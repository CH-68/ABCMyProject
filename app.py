import json
import pandas as pd
import streamlit as st
from crew import run_verification
from llm import answer_chat_question
from report import to_csv_bytes, to_word_bytes
from utility import extract_text
 

# ----------------------------
# App setup
# ----------------------------

st.set_page_config(page_title="Compliance Checker", page_icon="📄")
st.title("Compliance Checker")


# ----------------------------
# Cached PDF extraction
# ----------------------------

 
@st.cache_data(show_spinner=False)
def cached_extract_text(file_bytes: bytes) -> str:
    return extract_text(file_bytes)


def build_context(uploaded_files) -> str:
    if not uploaded_files:
        return ""

    parts = []
    for uploaded_file in uploaded_files:
        text = cached_extract_text(uploaded_file.getvalue())
        if text.strip():
            parts.append(f"[Document: {uploaded_file.name}]\n{text}")
    return "\n\n".join(parts).strip()

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

 
# ----------------------------
# Sidebar inputs
# ----------------------------

with st.sidebar:
    st.header("Upload PDFs")
    policy_files = st.file_uploader(
        "Policy Documents",
        type=["pdf"],
        accept_multiple_files=True,
    )
    user_files = st.file_uploader(
        "User Documents",
        type=["pdf"],
        accept_multiple_files=True,
    )
    run_clicked = st.button("Run Verification", type="primary")
 

# ----------------------------
# Orchestration
# ----------------------------

policy_context = build_context(policy_files)
user_context = build_context(user_files)

 
if run_clicked:
    if not policy_context or not user_context:
        st.error("Please upload both policy documents and user documents.")
    else:
        with st.spinner("Running verification..."):
            report = run_verification(policy_context, user_context)
            st.session_state["report"] = report
            st.success("Verification completed.")
 

# ----------------------------
# Results display
# ----------------------------

 
report = st.session_state.get("report")


if report:
    st.subheader("Final Verdict")
    c1, c2, c3 = st.columns(3)
    c1.metric("Verdict", report.get("verdict", ""))
    c2.metric("Covered", len(report.get("requirements_covered", [])))
    c3.metric("Missing", len(report.get("requirements_missing", [])))
    st.markdown(report.get("explanation", ""))


    issue_rows = []
    results_by_id = {r["issue_id"]: r for r in report.get("issue_results", [])}
    for issue in report.get("issue_inventory", []):
        result = results_by_id.get(issue["id"], {})
        issue_rows.append(
            {
                "Issue ID": issue.get("id", ""),
                "Title": issue.get("title", ""),
                "Target Rule": issue.get("target_rule", ""),
                "Verifier Status": result.get("verdict", ""),
                "Rationale": result.get("rationale", ""),
            }
        )

    if issue_rows:
        st.subheader("Issue Inventory")
        st.dataframe(pd.DataFrame(issue_rows), use_container_width=True, hide_index=True)
    # with st.expander("Evidence map"):
    #     st.json(report.get("evidence_map", {}))
    # with st.expander("Raw JSON"):
    #     st.json(report)
    st.subheader("Downloads")
    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "Download JSON",
        data=json.dumps(report, indent=2),
        file_name="compliance_report.json",
        mime="application/json",
    )
    col2.download_button(
        "Download CSV",
        data=to_csv_bytes(report),
        file_name="compliance_report.csv",
        mime="text/csv",
    )
    col3.download_button(
        "Download Word",
        data=to_word_bytes(report),
        file_name="compliance_report.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    st.divider()
    st.subheader("Chat with the verifier")
    st.caption("Ask follow-up questions about verification results.")

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    follow_up = st.chat_input("Ask a follow-up question")

    if follow_up:
        follow_up = follow_up.strip()
        if follow_up:
            st.session_state["chat_messages"].append(
                {"role": "user", "content": follow_up}
            )

            with st.spinner("Thinking..."):
                reply = answer_chat_question(
                    policy_context=policy_context,
                    user_context=user_context,
                    issue_inventory=report.get("issue_inventory", []),
                    issue_results=report.get("issue_results", []),
                    question=follow_up,
                    chat_history=st.session_state["chat_messages"],
                )

            st.session_state["chat_messages"].append(
                {"role": "assistant", "content": reply}
            )
            st.rerun()




else:
    st.info("Upload BOTH the policy and user documents, then run verification.")