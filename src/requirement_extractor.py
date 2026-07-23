import os
import streamlit as st
import json
#from dotenv import load_dotenv
from typing import Any
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# if load_dotenv('.env'):
#     # for local development
#     OPENAI_KEY = os.getenv('OPENAI_API_KEY')
# else:
OPENAI_KEY = st.secrets['OPENAI_API_KEY']
client = OpenAI(api_key=OPENAI_KEY)

def extract_policy_requirements(
    vectorstore_or_chunks: Any,
    top_k: int = 100,
) -> list[dict]:
    """
    Extract all policy requirements safely, whether given a Chroma vector store 
    or a raw list of Document chunks.
    """
    if not vectorstore_or_chunks:
        return []

    # 1. If a raw list of Documents was accidentally passed, handle it gracefully
    if isinstance(vectorstore_or_chunks, list):
        # Fallback option A: Create a temporary vector store on the fly so similarity_search works
        temp_store = Chroma.from_documents(
            documents=vectorstore_or_chunks,
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        )
        docs = temp_store.similarity_search(query="all policy requirements", k=top_k)
    
    # 2. If it's already a standard Chroma vector store
    elif hasattr(vectorstore_or_chunks, "similarity_search"):
        docs = vectorstore_or_chunks.similarity_search(
            query="all policy requirements",
            k=top_k,
        )
    else:
        docs = []

    if not docs:
        return []

    context = "\n\n".join(
        getattr(d, "page_content", str(d)) for d in docs
    )

    prompt = f"""
Extract EVERY individual compliance requirement.
Requirements:
• Preserve original wordings.
• Assign IDs:
R001
R002
...

Return ONLY JSON with a top-level key named "requirements".
Example:
{{
  "requirements": [
    {{
      "id": "R001",
      "text": "Passwords must contain at least 12 characters."
    }}
  ]
}}

Policy
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content
    obj = json.loads(content)

    # Handle cases where JSON root might be a list directly or wrapped under "requirements"
    if isinstance(obj, list):
        return obj
    return obj.get("requirements", [])