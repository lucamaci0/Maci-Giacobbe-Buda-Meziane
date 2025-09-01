from bs4 import BeautifulSoup
from crewai.tools import BaseTool
from doc_loader import DocLoaderTool
import json

def extract_template(html_path: str) -> dict:
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    print(html_content)

    soup = BeautifulSoup(html_content, "html.parser")
    schema = {}

    current_h1 = None
    current_h2 = None

    section = soup.find("div", class_="section")
    if section:
    # Cicla attraverso tutti i titoli e i <strong> nell'ordine in cui compaiono
        current_title = None
        for tag in section.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "strong"]):
            if tag.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                current_title = tag.get_text(strip=True)
                schema[current_title] = {}
            elif tag.name == "strong":
                if tag.next_sibling and str(tag.next_sibling).strip().startswith(":"):
                    schema[current_title][tag.get_text(strip=True)] = {}

    return schema

template = extract_template(r"C:\Users\XZ374JM\OneDrive - EY\Desktop\AI Academy\Maci-Giacobbe-Buda-Meziane\esercizio 29-08\rag_or_search\src\rag_or_search\tools\Application Documentation Template - techops.html")

print(json.dumps(template, indent=2, ensure_ascii=False))