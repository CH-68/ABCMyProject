import streamlit as st  
import random  
import hmac  
from langchain_core.documents import Document
# """  
# This file contains the common components used in the Streamlit App.  
# This includes the sidebar, the title, the footer, and the password check.  
# """  


def check_password():  
    """Returns `True` if the user had the correct password."""  
    def password_entered():  
        """Checks whether a password entered by the user is correct."""  
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):  
            st.session_state["password_correct"] = True  
            del st.session_state["password"]  # Don't store the password.  
        else:  
            st.session_state["password_correct"] = False  

    # Return True if the passward is validated.  
    if st.session_state.get("password_correct", False):  
        return True  

    # Show input for password.  
    st.text_input(  
        "Password", type="password", on_change=password_entered, key="password"  
    )  
    if "password_correct" in st.session_state:  
        st.error("Password incorrect")  
    return False


def safe_doc_to_context(doc: Document) -> str:
    meta = getattr(doc, "metadata", None)
    if not isinstance(meta, dict):
        meta = {}

    source = meta.get("source") or "Unknown Source"
    try:
        source = str(source).strip()
    except Exception:
        source = "Unknown Source"

    content = getattr(doc, "page_content", "") or ""
    try:
        content = str(content).strip()
    except Exception:
        content = ""

    return f"[Chunk | {source}] {content}".strip()

def format_docs(label: str, docs: list[Document]) -> str:
    if not docs:
        return f"{label}:\nNo relevant excerpts found."

    chunks = []
    for doc in docs:
        context = safe_doc_to_context(doc)
        if context:
            chunks.append(context)

    if not chunks:
        return f"{label}:\nNo relevant excerpts found."

    return f"{label}:\n" + "\n\n---\n\n".join(chunks)