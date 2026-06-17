"""Validation of the compiled wiki store schema and frontmatter."""

import re
import yaml
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field, ValidationError

class Violation(BaseModel):
    """Represents a single schema rule violation."""
    path: str
    rule: str
    message: str

class PageFrontmatter(BaseModel):
    """Pydantic model for page frontmatter validation."""
    title: str
    created: str
    updated: str
    sources: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

# Kebab case check for filenames (excluding .md)
KEBAB_CASE_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
# Wikilink pattern: [[page-name]]
WIKILINK_RE = re.compile(r"\[\[([a-z0-9-]+)\]\]")

def validate_page(path: str) -> List[Violation]:
    """Validates a single wiki page file against schema rules."""
    path_obj = Path(path)
    violations = []
    
    # 1. Naming validation
    filename_without_ext = path_obj.stem
    if not KEBAB_CASE_RE.match(filename_without_ext):
        violations.append(
            Violation(
                path=str(path_obj),
                rule="naming",
                message=f"Filename '{path_obj.name}' is not kebab-case."
            )
        )
    if path_obj.suffix != ".md":
        violations.append(
            Violation(
                path=str(path_obj),
                rule="naming",
                message=f"Filename '{path_obj.name}' does not have .md extension."
            )
        )
        return violations

    # 2. Read file content
    try:
        content = path_obj.read_text(encoding="utf-8")
    except Exception as e:
        violations.append(
            Violation(
                path=str(path_obj),
                rule="read",
                message=f"Failed to read file: {e}"
            )
        )
        return violations

    # 3. Parse YAML frontmatter
    if not (content.startswith("---\n") or content.startswith("---\r\n")):
        violations.append(
            Violation(
                path=str(path_obj),
                rule="frontmatter",
                message="File does not start with YAML frontmatter separator '---'."
            )
        )
        return violations
        
    parts = content.split("---", 2)
    if len(parts) < 3:
        violations.append(
            Violation(
                path=str(path_obj),
                rule="frontmatter",
                message="File is missing ending YAML frontmatter separator '---'."
            )
        )
        return violations
        
    frontmatter_raw = parts[1]
    body_content = parts[2]
    
    try:
        fm_dict = yaml.safe_load(frontmatter_raw) or {}
    except Exception as e:
        violations.append(
            Violation(
                path=str(path_obj),
                rule="frontmatter",
                message=f"Failed to parse YAML frontmatter: {e}"
            )
        )
        return violations

    # 4. Validate frontmatter fields
    try:
        PageFrontmatter(**fm_dict)
    except ValidationError as e:
        violations.append(
            Violation(
                path=str(path_obj),
                rule="frontmatter",
                message=f"Frontmatter validation failed: {e.errors()}"
            )
        )
        
    # 5. Check outgoing links
    links = WIKILINK_RE.findall(body_content)
    if not links:
        violations.append(
            Violation(
                path=str(path_obj),
                rule="links",
                message="Page has 0 outgoing links. Every page must have at least 1 outgoing link."
            )
        )
        
    return violations

def validate_vault(vault_dir: str) -> List[Violation]:
    """Validates the entire wiki vault against schema rules."""
    vault_path = Path(vault_dir)
    wiki_dir = vault_path / "wiki"
    
    if not wiki_dir.exists() or not wiki_dir.is_dir():
        return [
            Violation(
                path=str(wiki_dir),
                rule="directory",
                message="Wiki directory does not exist or is not a directory."
            )
        ]
        
    violations = []
    page_files = list(wiki_dir.glob("*.md"))
    existing_pages = {p.stem for p in page_files}
    
    for page_file in page_files:
        page_violations = validate_page(str(page_file))
        violations.extend(page_violations)
        
        # Check unresolved links
        try:
            content = page_file.read_text(encoding="utf-8")
            parts = content.split("---", 2)
            if len(parts) >= 3:
                body_content = parts[2]
                links = WIKILINK_RE.findall(body_content)
                for link in links:
                    if link not in existing_pages:
                        violations.append(
                            Violation(
                                path=str(page_file),
                                rule="unresolved_link",
                                message=f"Outgoing link [[{link}]] does not resolve to any page in the vault."
                            )
                        )
        except Exception:
            pass
            
    return violations
