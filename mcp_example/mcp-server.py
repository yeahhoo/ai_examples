from fastmcp import FastMCP
import os

DOCS_DIR = "./documents"
os.makedirs(DOCS_DIR, exist_ok=True)

# .venv\Scripts\activate
# deactivate 

mcp = FastMCP("document-server")

# ------------------------------
# Helpers
# ------------------------------

def get_doc_path(name: str):
    return os.path.join(DOCS_DIR, name)
	
@mcp.tool(
    name="find_document",
    description="Finds document in the given folder",
)
def find_document(name: str) -> list[str]:
    """Find documents by partial name match"""
    return [f for f in os.listdir(DOCS_DIR) if name.lower() in f.lower()]


@mcp.tool(
    name="view_document",
    description="Read the contents of a document and return it as a string.",
)
def view_document(name: str) -> str:
    """Return content of a document"""
    path = get_doc_path(name)
    if not os.path.exists(path):
        raise ValueError("Document not found")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()
		
@mcp.tool(
    name="create_document",
    description="Creates document with given name and content",
)
def create_document(name: str, content: str) -> str:
    """Overwrite document content"""
    path = get_doc_path(name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return content
	
@mcp.tool(
    name="get_document_list",
    description="Returns list of files",
)
def get_filenames() -> list[str]:
    return [
        f for f in os.listdir(DOCS_DIR)
        if os.path.isfile(os.path.join(DOCS_DIR, f))
    ]
	
@mcp.prompt(
    name="summarize_document",
    description="Summarize requested document by giving a short info what it's for",
)
def summarize_document(name: str) -> str:
    """Summarize a document"""
    content = view_document(name)
    return f"Summarize the following document:\n\n{content}"

@mcp.prompt(
    name="generate_document",
    description="Generate a document over requested topic",
)
def generate_document(topic: str) -> str:
    return f"Generate a random document with approximately 10 meaningful sentences over the subject: {topic}"

# @mcp.prompt()
# def improve_document(name: str) -> str:
#     """Improve writing quality"""
#     content = view_document(name)
#     return f"Improve clarity, grammar, and style:\n\n{content}"
#
# @mcp.prompt()
# def extract_todos(name: str) -> str:
#     """Extract TODO/action items"""
#     content = view_document(name)
#     return f"Extract all TODOs or action items:\n\n{content}"


# ------------------------------
# Run server
# ------------------------------

if __name__ == "__main__":
    # mcp.run()
	mcp.run(transport="http", port=8000)
