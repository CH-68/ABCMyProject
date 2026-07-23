import os
from dotenv import load_dotenv
import tempfile
import traceback
from typing import Any, Sequence
import pandas as pd
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import hashlib
from src.schemas import ComplianceReportSchema, ComplianceSummary
from helper_functions.utility import format_docs
from src.requirement_extractor import extract_policy_requirements
from src.crew import ComplianceCrew

# ----------------------------
# Setup
# ----------------------------

if load_dotenv('.env'):
    # for local development
    OPENAI_KEY = os.getenv('OPENAI_API_KEY')
else:
    OPENAI_KEY = st.secrets['OPENAI_API_KEY']
client = OpenAI(api_key=OPENAI_KEY)

os.environ["CREWAI_TRACING"] = "true"

def normalize_uploaded_files(uploaded_files: Any) -> list[Any]:
    """Accept either a single uploaded file or a list of uploaded files."""
    if uploaded_files is None:
        return []
    if isinstance(uploaded_files, (list, tuple)):
        return list(uploaded_files)
    return [uploaded_files]

# ----------------------------
# RAG: Load + Split PDF
# ----------------------------
def load_and_split(uploaded_files: Sequence[Any]) -> list[Document]:
    """Load one or more uploaded PDFs, split them into chunks, and return LangChain Document chunks."""
    files = normalize_uploaded_files(uploaded_files)
    if not files:
        raise ValueError("No files were uploaded.")

    all_chunks: list[Document] = []

    for uploaded_file in files:
        data = uploaded_file.read()
        if not data:
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            temp_file_path = tmp.name

        try:
            loader = PyPDFLoader(temp_file_path)
            pages = loader.load()

            if not pages:
                continue

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", " ", ""],
            )
            chunks = text_splitter.split_documents(pages)

            for chunk in chunks:
                metadata = dict(getattr(chunk, "metadata", {}) or {})
                metadata["source"] = getattr(uploaded_file, "name", "uploaded.pdf")
                chunk.metadata = metadata

            all_chunks.extend(chunks)
        except Exception as e:
            st.error(f"Error splitting document {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    if not all_chunks:
        raise ValueError("No readable text could be extracted from the uploaded PDFs.")

    return all_chunks


# ----------------------------
# RAG: Build Vector Store
# ----------------------------
def file_fingerprint(uploaded_files: Sequence[Any]) -> str:
    """Stable-ish id for the uploaded content."""
    files = normalize_uploaded_files(uploaded_files)
    entries = []

    for uploaded_file in files:
        name = getattr(uploaded_file, "name", None) or "uploaded.pdf"
        size = getattr(uploaded_file, "size", None)
        mtime = getattr(uploaded_file, "last_modified", None)
        entries.append(f"{name}|{size}|{mtime}")
    raw = "|".join(entries).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


@st.cache_resource(max_entries=10, show_spinner=False)
def build_vectorstore(chunks: list[Document], collection_name: str) -> Chroma:
    return Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name=collection_name,
    )

# ----------------------------
# RAG: System Prompt
# ----------------------------
def build_grounded_rag_system_prompt(policy_context: str, user_context: str) -> str:
    fallback = "I couldn't find that information in the uploaded documents."

    return f"""You are a senior compliance analyst.
Your task is to compare the User Document against the Policy Document, using the Policy Document as the master standard.
Instructions:
1. Use the Policy Document as the authoritative baseline.
2. Compare the User Document against it and identify any gaps, mismatches, policy violations, or missing requirements.
3. Structure your answer as: finding, evidence, confidence level.
4. Include short citations like [Page 2] or [Section 2.2.3] when available in the provided context.
5. Use only the provided context. Do not use outside facts or assumptions.
6. If the context is insufficient, say: "{fallback}"

Policy Document Context:
-----------------
{policy_context}
-----------------
User Document Context:
-----------------
{user_context}
-----------------
Now answer the user's question by comparing the User Document to the Policy Document using only the context above."""

# ----------------------------
# RAG: Retrieve Context
# ----------------------------
def retrieve_context(vectorstore: Chroma, query: str, k: int, score_threshold: float):
    if query is None:
        return None
    query = str(query).strip()
    if not query:
        return None

    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold},
        )
        results = retriever.invoke(query)
        if not results:
            st.warning(
                "No relevant context found matching the query and current retrieval settings."
            )
            return None
        return results
    except Exception as e:
        st.error(f"Error during document retrieval (Check parameters/API): {e}")
        traceback.print_exc()
        return None

# ----------------------------
# 1. INITIALIZE SESSION STATE KEYS
# ----------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("k_value", 10)
st.session_state.setdefault("score_threshold", 0.01)
st.session_state.setdefault("policy_vectorstore", None)
st.session_state.setdefault("user_vectorstore", None)
st.session_state.setdefault("policy_fingerprint", None)
st.session_state.setdefault("policy_requirements", [])
st.session_state.setdefault("user_fingerprint", None)
st.session_state.setdefault("compliance_crew", None)

if "k_value" not in st.session_state:
    st.session_state["k_value"] = 10

if "score_threshold" not in st.session_state:
    st.session_state["score_threshold"] = 0.01

# ----------------------------
# Streamlit UI Setup
# ----------------------------
st.set_page_config(page_title="Compliance Verifier (RAG Grounded)")
st.title("Compliance Verifier")

# ----------------------------
# Streamlit UI: Sidebar & Document Persistence
# ----------------------------
with st.sidebar:
    st.header("📂 Knowledge Bases")
    st.markdown("**Policy Document**")
    policy_files = st.file_uploader(
        "Upload Reference PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key="policy_uploader",
    )

    st.markdown("---")
    st.subheader("**User Document**")
    user_files = st.file_uploader(
        "Upload PDF to verify",
        type=["pdf"],
        accept_multiple_files=True,
        key="user_uploader",
    )

    # Policy processing
    if policy_files:
        try:
            policy_hash = file_fingerprint(policy_files)
            if st.session_state.policy_fingerprint != policy_hash:
                with st.spinner("Processing Policy Documents..."):
                    # 1. Load and split chunks
                    policy_chunks = load_and_split(policy_files)
                    st.session_state.policy_chunks = policy_chunks
                    
                    # 2. Build the vector store FIRST so it's a Chroma object
                    policy_name = f"chroma_policy_{policy_hash}"
                    policy_vs = build_vectorstore(policy_chunks, policy_name)
                    st.session_state.policy_vectorstore = policy_vs
                    
                    # 3. Pass the Chroma vector store into the requirements extractor
                    st.session_state.policy_requirements = extract_policy_requirements(
                        policy_vs
                    )
                    
                    st.session_state.policy_fingerprint = policy_hash
        except Exception as e:
            st.error(f"Failed to load Policy PDFs: {e}")

    # User processing
    if user_files:
        try:
            user_hash = file_fingerprint(user_files)
            if st.session_state.user_fingerprint != user_hash:
                with st.spinner("Processing User Documents..."):
                    user_chunks = load_and_split(user_files)
                    st.session_state.user_chunks = user_chunks
                    user_name = f"chroma_user_{user_hash}"
                    st.session_state.user_vectorstore = build_vectorstore(
                        user_chunks, user_name
                    )
                    st.session_state.user_fingerprint = user_hash
        except Exception as e:
            st.error(f"Failed to load User PDFs: {e}")

    # Instantiate ComplianceCrew only when both stores are ready
    if st.session_state.policy_vectorstore and st.session_state.user_vectorstore:
        if st.session_state.get("compliance_crew") is None:
            st.session_state.compliance_crew = ComplianceCrew(
                policy_vectorstore=st.session_state.policy_vectorstore,
                user_vectorstore=st.session_state.user_vectorstore,
            )
        st.success("Both documents processed.")   

    # Run Verification inside Sidebar
    st.markdown("---")
    st.subheader("Verification")
    if st.button("Run Verification"):
        if not st.session_state.get("compliance_crew"):
            st.error("ComplianceCrew is not initialized properly.")
        else:
            with st.spinner("Running semantic agentic verification..."):
                try:
                    crew_instance = st.session_state.compliance_crew
                    findings = []
                    
                    for requirement in st.session_state.policy_requirements:
                        finding = crew_instance.verify_requirement(requirement)                
                        if finding:
                            findings.extend(finding.findings)

                    # Example Summary calculation based on findings
                    passed_count = sum(1 for f in findings if getattr(f, "verification", "") == "passed")
                    failed_count = sum(1 for f in findings if getattr(f, "verification", "") == "failed")
                    deviated_count = sum(1 for f in findings if getattr(f, "verification", "") == "deviated")
                    unknown_count = sum(1 for f in findings if getattr(f, "verification", "") == "unknown")
                    
                    summary = ComplianceSummary(
                        total_requirements=len(st.session_state.policy_requirements),
                        passed=passed_count,
                        failed=failed_count,
                        deviated=deviated_count,
                        unknown=unknown_count
                    )
                    
                    report = ComplianceReportSchema(
                        summary=summary,
                        findings=findings,
                    )
                    st.session_state["compliance_report"] = report
                    st.success("Verification completed successfully!")

                except Exception as e:
                    st.exception(e)
                    st.code(traceback.format_exc())

# ----------------------------
# Streamlit UI: Verification Results (Main Area)
# ----------------------------
compliance_report = st.session_state.get("compliance_report")

if compliance_report:
    st.subheader("Verification Summary")
    summary = compliance_report.summary
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Requirements", summary.total_requirements)
    with col2:
        st.metric("Passed", summary.passed)
    with col3:
        st.metric("Deviated", summary.deviated)
    with col4:
        st.metric("Failed", summary.failed)
    with col5:
        st.metric("Unknown", summary.unknown)

    st.subheader("Verification Findings")
    findings = [
        finding
        for finding in compliance_report.findings
        if finding.verification in ("deviated", "failed","unknown")
    ]

    if findings:
        findings_df = pd.DataFrame([f.model_dump() for f in findings])
        st.dataframe(
            findings_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("✓ All requirements passed. No policy deviations or failures were identified.")

# ----------------------------
# Streamlit Chat (Main Area)
# ----------------------------
st.markdown("---")
st.markdown("#### Upload and Run Verification, then Query Results")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Query findings here")

if user_input:
    user_input = user_input.strip()
    st.session_state["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    policy_vectorstore = st.session_state.get("policy_vectorstore")
    user_vectorstore = st.session_state.get("user_vectorstore")

    if not policy_vectorstore or not user_vectorstore:
        st.warning("Please upload both Policy PDFs and User PDFs to compare them.")
        st.stop()

    with st.spinner("Searching both knowledge bases and generating response..."):
        k_value = st.session_state["k_value"]
        policy_docs = policy_vectorstore.similarity_search(user_input, k=k_value)
        user_docs = user_vectorstore.similarity_search(user_input, k=k_value)

        policy_context = "\n\n".join([doc.page_content for doc in policy_docs])
        user_context = "\n\n".join([doc.page_content for doc in user_docs])

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a compliance assistant.
                    Answer only using the supplied policy and user document context.
                    Make cross reference to Compliance Report and elaborate or explain in more details.
                    If the answer cannot be found, say so explicitly.
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    Question:
                    {user_input}
                    Policy Context:
                    {policy_context}
                    User Document Context:
                    {user_context}
                    Compliance Report:
                    (compliance_report)
                    """
                }
            ]
        )

        response_content = response.choices[0].message.content
        st.session_state["messages"].append(
            {"role": "assistant", "content": response_content}
        )
        with st.chat_message("assistant"):
            st.write(response_content)