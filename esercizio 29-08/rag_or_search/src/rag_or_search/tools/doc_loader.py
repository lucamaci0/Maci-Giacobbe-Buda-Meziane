"""Documentation loader tool for CrewAI agents.

This tool reads documentation files from the docs folder and returns them as structured JSON.
Supports HTML, RST, and other text-based documentation formats.
"""

import json
import os
from pathlib import Path
from typing import Type, Dict, List, Any, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import re


class DocLoaderInput(BaseModel):
    """Input schema for DocLoaderTool.

    Parameters
    ----------
    docs_path : str, optional
        Path to the documentation folder. Defaults to "docs".
    file_types : List[str], optional
        List of file extensions to include (ignored, only reads index.html). Defaults to [".html"].
    include_metadata : bool, optional
        Whether to include file metadata in the output. Defaults to True.
    max_file_size : int, optional
        Maximum file size to process in bytes. Defaults to 1MB.
    """
    docs: str = Field(default="docs", description="Use 'docs' to read the documentation, use 'template' to read the AI Act template")
    file_types: List[str] = Field(default=[".html"], description="List of file extensions to include (ignored, only reads index.html).")
    include_metadata: bool = Field(default=True, description="Whether to include file metadata in the output.")
    max_file_size: int = Field(default=1048576, description="Maximum file size to process in bytes (1MB default).")


class DocLoaderTool(BaseTool):
    """CrewAI tool that reads the index.html documentation file and returns it as structured JSON.

    This tool specifically reads the index.html file from the docs/_build/html/ directory.
    It extracts text content, preserves structure, and returns everything as a JSON object
    with metadata about the file.
    """

    name: str = "Documentation Loader Tool"
    description: str = (
        "A tool to read the index.html documentation file from docs/_build/html/ and return it "
        "as structured JSON. Extracts text content while preserving document structure and metadata."
    )
    args_schema: Type[BaseModel] = DocLoaderInput

    def _extract_html_content(self, html_content: str, file_path: str) -> Dict[str, Any]:
        """Extract structured content from HTML files.
        
        Args
        ----
        html_content : str
            The HTML content to parse.
        file_path : str
            Path to the HTML file for metadata.
            
        Returns
        -------
        dict
            Structured content with sections, headings, and text.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ""
        
        # Extract main content
        main_content = soup.find('div', class_='body') or soup.find('main') or soup.find('body')
        
        content = {
            "title": title_text,
            "sections": [],
            "raw_text": "",
            "links": [],
            "images": []
        }
        
        if main_content:
            # Extract headings and sections
            headings = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            for heading in headings:
                section = {
                    "level": int(heading.name[1]),
                    "text": heading.get_text().strip(),
                    "id": heading.get('id', ''),
                    "content": ""
                }
                
                # Get content until next heading of same or higher level
                next_element = heading.next_sibling
                while next_element:
                    if hasattr(next_element, 'name'):
                        if next_element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            if int(next_element.name[1]) <= section["level"]:
                                break
                        section["content"] += next_element.get_text() + "\n"
                    else:
                        section["content"] += str(next_element)
                    next_element = next_element.next_sibling
                
                content["sections"].append(section)
            
            # Extract links
            links = main_content.find_all('a', href=True)
            content["links"] = [{"text": link.get_text().strip(), "href": link['href']} for link in links]
            
            # Extract images
            images = main_content.find_all('img')
            content["images"] = [{"alt": img.get('alt', ''), "src": img.get('src', '')} for img in images]
            
            # Get raw text
            content["raw_text"] = main_content.get_text()
        
        return content

    def _extract_rst_content(self, rst_content: str, file_path: str) -> Dict[str, Any]:
        """Extract structured content from RST files.
        
        Args
        ----
        rst_content : str
            The RST content to parse.
        file_path : str
            Path to the RST file for metadata.
            
        Returns
        -------
        dict
            Structured content with sections and text.
        """
        lines = rst_content.split('\n')
        content = {
            "title": "",
            "sections": [],
            "raw_text": rst_content,
            "directives": []
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Extract title (first non-empty line that's not a directive)
            if not content["title"] and line and not line.startswith('..'):
                content["title"] = line
            
            # Extract headings (lines with only =, -, or other characters)
            if line and all(c in '=-~`+*^"\'#<>' for c in line) and len(line) > 3:
                if current_section:
                    content["sections"].append(current_section)
                current_section = {
                    "level": 1 if '=' in line else 2 if '-' in line else 3,
                    "text": "",
                    "content": ""
                }
            elif current_section and not line.startswith('..'):
                if not current_section["text"] and line:
                    current_section["text"] = line
                else:
                    current_section["content"] += line + "\n"
            
            # Extract directives
            if line.startswith('..'):
                content["directives"].append(line)
        
        if current_section:
            content["sections"].append(current_section)
        
        return content

    def _extract_md_content(self, md_content: str, file_path: str) -> Dict[str, Any]:
        """Extract structured content from Markdown files.
        
        Args
        ----
        md_content : str
            The Markdown content to parse.
        file_path : str
            Path to the Markdown file for metadata.
            
        Returns
        -------
        dict
            Structured content with sections and text.
        """
        lines = md_content.split('\n')
        content = {
            "title": "",
            "sections": [],
            "raw_text": md_content,
            "links": [],
            "images": []
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Extract title (first # heading)
            if not content["title"] and line.startswith('# '):
                content["title"] = line[2:].strip()
            
            # Extract headings
            if line.startswith('#'):
                if current_section:
                    content["sections"].append(current_section)
                
                level = len(line) - len(line.lstrip('#'))
                heading_text = line.lstrip('#').strip()
                current_section = {
                    "level": level,
                    "text": heading_text,
                    "content": ""
                }
            elif current_section:
                current_section["content"] += line + "\n"
            
            # Extract links [text](url)
            link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
            links = re.findall(link_pattern, line)
            content["links"].extend([{"text": text, "href": url} for text, url in links])
            
            # Extract images ![alt](src)
            img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
            images = re.findall(img_pattern, line)
            content["images"].extend([{"alt": alt, "src": src} for alt, src in images])
        
        if current_section:
            content["sections"].append(current_section)
        
        return content

    def _get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get metadata for a file.
        
        Args
        ----
        file_path : Path
            Path to the file.
            
        Returns
        -------
        dict
            File metadata including size, modification time, etc.
        """
        stat = file_path.stat()
        return {
            "name": file_path.name,
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "extension": file_path.suffix,
            "relative_path": str(file_path)
        }

    def _run(self, docs: str = "docs", file_types: List[str] = None, 
             include_metadata: bool = True, max_file_size: int = 1048576) -> str:
        """Load documentation files and return as JSON.

        Args
        ----
        docs : str
            Use 'docs' to read the documentation, use 'template' to read the AI Act template
        file_types : List[str]
            List of file extensions to include (ignored, only reads index.html).
        include_metadata : bool
            Whether to include file metadata in the output.
        max_file_size : int
            Maximum file size to process in bytes.

        Returns
        -------
        str
            JSON string containing all documentation content.

        Raises
        ------
        FileNotFoundError
            If the index.html file doesn't exist.
        ValueError
            If the index.html file cannot be processed.
        """
        if docs == "docs":
            docs_path = Path(r"C:\Users\XZ374JM\OneDrive - EY\Desktop\AI Academy\Maci-Giacobbe-Buda-Meziane\esercizio 29-08\rag_or_search\docs")
            index_file = docs_path / "_build" / "html" / "index.html"
        elif docs == "template":
            index_file = Path(r"C:\Users\XZ374JM\OneDrive - EY\Desktop\AI Academy\Maci-Giacobbe-Buda-Meziane\esercizio 29-08\rag_or_search\Application Documentation Template - techops.html")    
 
        if not index_file.exists():
            raise FileNotFoundError(f"Index file '{index_file}' does not exist.")
        
        result = {
            "metadata": {
                "docs_path": str(docs_path if docs == "docs" else index_file.parent),
                "file_types": [".html"],
                "total_files": 1,
                "processed_files": 0,
                "skipped_files": 0
            },
            "files": []
        }
        
        try:
            # Check file size
            if index_file.stat().st_size > max_file_size:
                result["metadata"]["skipped_files"] = 1
                raise ValueError(f"Index file '{index_file}' exceeds maximum size limit of {max_file_size} bytes.")
            
            # Read file content
            with open(index_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract content
            file_data = {
                "path": str(index_file),
                "type": ".html",
                "content": {}
            }
            
            if include_metadata:
                file_data["metadata"] = self._get_file_metadata(index_file)
            
            # Extract HTML content
            file_data["content"] = self._extract_html_content(content, str(index_file))
            
            result["files"].append(file_data)
            result["metadata"]["processed_files"] = 1
            
        except Exception as e:
            print(f"Error processing file {index_file}: {str(e)}")
            result["metadata"]["skipped_files"] = 1
            raise ValueError(f"Failed to process index.html: {str(e)}")
        
        return json.dumps(result, indent=2, ensure_ascii=False)
