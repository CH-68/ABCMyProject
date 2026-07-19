from io import BytesIO
from pathlib import Path
from typing import Any
from pypdf import PdfReader

# ----------------------------
# Access control
# ----------------------------

# def check_password():  
#     """Returns `True` if the user had the correct password."""  
#     def password_entered():  
#         """Checks whether a password entered by the user is correct."""  
#         if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):  
#             st.session_state["password_correct"] = True  
#             del st.session_state["password"]  # Don't store the password.  
#         else:  
#             st.session_state["password_correct"] = False  

#     # Return True if the passward is validated.  
#     if st.session_state.get("password_correct", False):  
#         return True  

#     # Show input for password.  
#     st.text_input(  
#         "Password", type="password", on_change=password_entered, key="password"  
#     )  
#     if "password_correct" in st.session_state:  
#         st.error("😕 Password incorrect")  
#     return False


# ----------------------------
# PDF text extraction
# ----------------------------

def extract_text(file: Any) -> str:
    """
    Read PDF text from a path, bytes, uploaded file, or file-like object.
    Returns page-marked text so the verifier can point to evidence cleanly.
    """
    if file is None:
        return ""

    # Resolve the input into a PdfReader
    if isinstance(file, (str, Path)):
        reader = PdfReader(str(file))
    else:
        if isinstance(file, (bytes, bytearray)):
            stream = BytesIO(file)
        elif hasattr(file, "getvalue"):
            stream = BytesIO(file.getvalue())
        elif hasattr(file, "read"):
            data = file.read()
            if hasattr(file, "seek"):
                file.seek(0)
            stream = BytesIO(data)
        else:
            raise TypeError("extract_text() expects a PDF path, bytes, or file-like object.")

        reader = PdfReader(stream)

    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"[Page {page_number}]\n{text}")

    return "\n\n".join(pages).strip()