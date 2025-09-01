from bs4 import BeautifulSoup
from crewai.tools import BaseTool
 
class TemplateLoaderTool(BaseTool):
    """Parses AI Act template from HTML and outputs a nested JSON schema."""
 
    name : str = "AI Act Template Loader"
    description : str = "Loads and parses an AI Act compliance template from HTML into a structured schema."
 
    def _run(self, html_path: str) -> dict:
        html_path = r"C:\Users\XZ374JM\OneDrive - EY\Desktop\AI Academy\Maci-Giacobbe-Buda-Meziane\esercizio 29-08\rag_or_search\src\rag_or_search\tools\Application Documentation Template - techops.html"
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
 
        soup = BeautifulSoup(html_content, "html.parser")
        schema = {}
 
        current_h1 = None
        current_h2 = None
 
        for tag in soup.find_all(["h1", "h2", "h3"]):
            if tag.name == "h1":
                current_h1 = tag.get_text(strip=True)
                schema[current_h1] = {}
            elif tag.name == "h2" and current_h1:
                current_h2 = tag.get_text(strip=True)
                schema[current_h1][current_h2] = {}
            elif tag.name == "h3" and current_h1 and current_h2:
                schema[current_h1][current_h2][tag.get_text(strip=True)] = (
                    "[PLACEHOLDER: Fill in details for this section]"
                )
 
        return schema