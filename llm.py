import json
import os
from functools import lru_cache
from typing import Any, Dict, List
from openai import OpenAI
from schemas import Issue

# ----------------------------
# Client setup
# ----------------------------
def get_completion(
        messages: List[Dict[str,str]],
        model: str ="gpt-4o-mini",
        temperature: float = 0.1,
) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception as exc:
            raise RuntimeError("OPENAI_API_KEY is missing.") from exc
    return OpenAI(api_key=api_key)
 

def _json_completion(messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> Dict[str, Any]:
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "{}"
    return json.loads(raw)

 

# ----------------------------
# Issue generation
# ----------------------------
 
def generate_issues(policy_context: str, user_context: str) -> Dict[str, Any]:
    """
    GPT produces a short summary plus a small issue inventory.
    Returns:
        {
            "summary": "...",
            "issues": [Issue(...), ...]
        }
    """
    system_prompt = """
    You are a compliance issue generator.

    Task:
    - Compare the policy context and user context.
    - Return a SHORT summary plus a SMALL inventory of candidate issues.
    - Keep the list focused and auditable.
    Return ONLY JSON in this shape:

    {
      "summary": "short paragraph",
      "issues": [
        {
          "title": "short issue title",
          "summary": "1-2 sentence explanation",
          "target_rule": "the policy rule or requirement being checked",
            "evidence_pointers": ["[Page 2]", "[Page 5]"]
        }
    ]
    }

    Rules:
    - ONLY issues that verify_issue() returns fail or insufficient.
    - List speific issues that are evidence-based, not vague or arguable ones.
    - Use only the provided context as evidence.
    - If nothing looks notable, return an empty issues array.
    """.strip()

    payload = _json_completion(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Policy context:\n"
                    f"{policy_context}\n\n"
                    "User context:\n"
                    f"{user_context}"
                ),
            },
        ]
    )

 
    summary = str(payload.get("summary", "")).strip()
    issues_raw = payload.get("issues", []) or []

    issues: list[Issue] = []
    for item in issues_raw[:5]:
        if not isinstance(item, dict):
            continue
        issues.append(
            Issue(
                title=str(item.get("title", "")).strip(),
                summary=str(item.get("summary", "")).strip(),
                target_rule=str(item.get("target_rule", "")).strip(),
                evidence_pointers=[
                    str(x).strip() for x in (item.get("evidence_pointers") or []) if str(x).strip()
                ],
            )
        )
    return {"summary": summary, "issues": issues}

# ----------------
# Chatbot to clarify verification results
# ----------------

def answer_chat_question(
    policy_context: str,
    user_context: str,
    issue_inventory: List[Dict[str, Any]],
    issue_results: List[Dict[str, Any]],
    question: str,
    chat_history: List[Dict[str, str]] | None = None,
    ) -> str:
    system_prompt = """
    You are a compliance assistant inside a document review app.
    Use the policy_context and user_context as primary sources.
    Use issue_inventory and issue_results as the verified reference layer.
    Do not contradict the verified findings.
    If the question goes beyond the available evidence, fall back on policy_context and user_context.
    Explain why they were flag in issue_inventory and issue_results or left out of the verified findings.
    When relevant, mention issue IDs and whether the evidence supports, fails, or is insufficient.
    Be concise and grounded.
    """.strip()

    report_context = json.dumps(
        {
        "issue_inventory": issue_inventory,
        "issue_results": issue_results,
        },
        indent=2,
    )
 
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if chat_history:
        for msg in chat_history[-6:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
    messages.append(
        {
        "role": "user",
        "content": (
                f"Policy context:\n{policy_context}\n\n"
                f"User context:\n{user_context}\n\n"
                f"Verified report context:\n{report_context}\n\n"
                f"User question:\n{question}"
            ),
        }
    )
    return get_completion(messages, temperature=0.1)