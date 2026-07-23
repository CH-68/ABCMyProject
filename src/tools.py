from typing import Any, List
from pydantic import ConfigDict
from crewai.tools import BaseTool
from langchain_core.documents import Document
from helper_functions.utility import format_docs

def _retrieve(vectorstore_or_docs: Any, query: str) -> List[Document]:
    """Safely handles retrieval whether vectorstore is a Chroma instance or a list."""
    if not vectorstore_or_docs:
        return []

    # Fallback guard if a raw list of documents was accidentally passed
    if isinstance(vectorstore_or_docs, list):
        return [doc for doc in vectorstore_or_docs if isinstance(doc, Document)]

    # Standard Chroma vector store retriever invocation
    if hasattr(vectorstore_or_docs, "as_retriever"):
        try:
            retriever = vectorstore_or_docs.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 10,
                    "score_threshold": 0.5,
                },
            )
            return retriever.invoke(query) or []
        except Exception:
            pass

    return []


class PolicySearchTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "Search Policy Document"
    description: str = (
        "Search for specific rules, authoritative requirements, or compliance standards in the Policy Document. "
        "Input must be a descriptive natural language query asking about a specific policy rule or standard."
    )
    vectorstore: Any

    def _run(self, query: str) -> str:
        docs = _retrieve(self.vectorstore, query)
        return format_docs("Policy Document", docs)


class UserDocSearchTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "Search User Document"
    description: str = (
        "Search for clauses, statements, or implementation details inside the User Document to verify against policy. "
        "Input must be a descriptive natural language query asking about how a requirement is addressed."
    )
    vectorstore: Any

    def _run(self, query: str) -> str:
        docs = _retrieve(self.vectorstore, query)
        return format_docs("User Document", docs)