import hashlib
import json
from typing import Any, Dict, List
from llm import generate_issues, _json_completion
from schemas import ComplianceReportSchema, Issue, IssueVerification


# ----------------------------
# Stable IDs
# ----------------------------


def _stable_issue_id(issue: Issue) -> str:
    seed = f"{issue.title}|{issue.target_rule}|{issue.summary}".strip().lower()
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def _normalize_issues(raw_issues: List[Issue]) -> List[Issue]:
    normalized: list[Issue] = []
    for issue in raw_issues:
        issue.id = issue.id or _stable_issue_id(issue)
        normalized.append(issue)
    return normalized


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    output: list[str] = []
    for item in items:
        value = str(item).strip()
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


# ----------------------------
# Independent issue verification
# ----------------------------


def verify_issue(issue: Dict[str, Any], policy_context: str, user_context: str) -> Dict[str, Any]:
    """
    Verify ONE issue only.
    Returns pass / fail / insufficient without affecting the rest of the review.
    """
    system_prompt = """
You are a compliance verifier.
You review exactly one issue at a time.
Use only the provided policy context and user context.
Do not broaden scope and do not rely on other issues.
Return ONLY JSON in this shape:
{
  "verdict": "pass|fail|insufficient",
  "rationale": "short explanation",
  "evidence": ["short evidence pointers or brief excerpts"],
  "suggested_fix": "brief fix when relevant"
}
Decision rules:
- pass: the contexts support that the issue is satisfied.
- fail: the contexts support that the issue is violated or missing.
- insufficient: the contexts do not contain enough proof either way.
""".strip()
    issue_block = json.dumps(issue, indent=2)

    try:
        payload = _json_completion(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Issue:\n{issue_block}\n\n"
                        f"Policy context:\n{policy_context}\n\n"
                        f"User context:\n{user_context}"
                    ),
                },
            ]
        )
    except Exception:
        payload = {}

    verdict = str(payload.get("verdict", "insufficient")).strip().lower()
    if verdict not in {"pass", "fail", "insufficient"}:
        verdict = "insufficient"

    evidence = [
        str(x).strip() for x in (payload.get("evidence") or []) if str(x).strip()
    ]

    return {
        "issue_id": str(issue.get("id", "")).strip(),
        "verdict": verdict,
        "rationale": str(payload.get("rationale", "")).strip(),
        "evidence": evidence,
        "suggested_fix": str(payload.get("suggested_fix", "")).strip(),
    }


# ----------------------------
# One-pass orchestration
# ----------------------------

def run_verification(policy_context: str, user_context: str) -> Dict[str, Any]:
    """
    Generate issues once, verify each issue independently once,
    then return a plain dictionary matching ComplianceReportSchema.
    """
    issue_bundle = generate_issues(policy_context, user_context)
    summary = str(issue_bundle.get("summary", "")).strip()
    issues = _normalize_issues(issue_bundle.get("issues", []) or [])
    results: list[IssueVerification] = []
    evidence_map: dict[str, list[str]] = {}
    citations: list[str] = []
    requirements_covered: list[str] = []
    requirements_missing: list[str] = []

    for issue in issues:
        result_dict = verify_issue(
            issue.model_dump(), policy_context, user_context)
        result = IssueVerification(**result_dict)
        results.append(result)

        issue_evidence = _dedupe_preserve_order(
            list(issue.evidence_pointers) + list(result.evidence)
        )
        evidence_map[issue.id] = issue_evidence
        citations.extend(issue_evidence)

        label = issue.target_rule or issue.title or issue.id
        if result.verdict == "pass":
            requirements_covered.append(label)
        elif result.verdict == "fail":
            requirements_missing.append(label)

    citations = _dedupe_preserve_order(citations)

    if any(r.verdict == "fail" for r in results):
        final_verdict = "failed"
    elif any(r.verdict == "insufficient" for r in results):
        final_verdict = "partially_cleared"
    else:
        final_verdict = "fully_cleared"

    explanation = (
        f"Generated {len(issues)} candidate issue(s) and verified each one independently."
    )

    if summary:
        explanation = f"{summary}\n\n{explanation}"
        report = ComplianceReportSchema(
            verdict=final_verdict,
            summary=summary,
            issue_inventory=issues,
            issue_results=results,
            evidence_map=evidence_map,
            # requirements_covered=_dedupe_preserve_order(requirements_covered),
            # requirements_missing=_dedupe_preserve_order(requirements_missing),
            citations=citations,
            explanation=explanation,
        )
    return report.model_dump()
