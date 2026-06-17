"""Unit tests for the Path B wiki schema validation logic."""

import pytest
from pathlib import Path
from rem.wiki.schema import validate_page, validate_vault, Violation

def test_validate_page_valid(tmp_path):
    page_file = tmp_path / "valid-page.md"
    content = """---
title: Valid Page
created: "2026-06-10T12:00:00Z"
updated: "2026-06-10T12:00:00Z"
sources:
  - "raw/source1.txt"
tags:
  - test
---
This is a valid page with an outgoing [[link-target]].
"""
    page_file.write_text(content, encoding="utf-8")
    
    violations = validate_page(str(page_file))
    assert len(violations) == 0

def test_validate_page_invalid_name(tmp_path):
    page_file = tmp_path / "InvalidName.md"
    content = """---
title: Invalid Name
created: "2026-06-10T12:00:00Z"
updated: "2026-06-10T12:00:00Z"
sources: []
tags: []
---
This has an outgoing [[link]].
"""
    page_file.write_text(content, encoding="utf-8")
    
    violations = validate_page(str(page_file))
    assert len(violations) == 1
    assert violations[0].rule == "naming"
    assert "is not kebab-case" in violations[0].message

def test_validate_page_invalid_extension(tmp_path):
    page_file = tmp_path / "valid-page.txt"
    content = """---
title: Valid Page
created: "2026-06-10T12:00:00Z"
updated: "2026-06-10T12:00:00Z"
sources: []
tags: []
---
This is a valid page with [[link]].
"""
    page_file.write_text(content, encoding="utf-8")
    
    violations = validate_page(str(page_file))
    assert len(violations) >= 1
    assert violations[0].rule == "naming"
    assert "does not have .md extension" in violations[0].message

def test_validate_page_missing_frontmatter(tmp_path):
    page_file = tmp_path / "missing-frontmatter.md"
    content = """This is missing frontmatter completely.
It has a [[link]] though.
"""
    page_file.write_text(content, encoding="utf-8")
    
    violations = validate_page(str(page_file))
    assert len(violations) >= 1
    assert violations[0].rule == "frontmatter"

def test_validate_page_invalid_frontmatter_fields(tmp_path):
    page_file = tmp_path / "invalid-fields.md"
    # Missing 'created' and 'updated'
    content = """---
title: Page Title
sources: []
tags: []
---
This is a page with a [[link]].
"""
    page_file.write_text(content, encoding="utf-8")
    
    violations = validate_page(str(page_file))
    assert len(violations) == 1
    assert violations[0].rule == "frontmatter"
    assert "validation failed" in violations[0].message

def test_validate_page_missing_links(tmp_path):
    page_file = tmp_path / "no-links.md"
    content = """---
title: No Links Page
created: "2026-06-10T12:00:00Z"
updated: "2026-06-10T12:00:00Z"
sources: []
tags: []
---
This page contains no outgoing links whatsoever.
"""
    page_file.write_text(content, encoding="utf-8")
    
    violations = validate_page(str(page_file))
    assert len(violations) == 1
    assert violations[0].rule == "links"
    assert "0 outgoing links" in violations[0].message

def test_validate_vault_resolves_correctly(tmp_path):
    # Setup vault skeleton
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    
    page1 = wiki_dir / "page-one.md"
    page1_content = """---
title: Page One
created: "2026-06-10T12:00:00Z"
updated: "2026-06-10T12:00:00Z"
sources: []
tags: []
---
Link to [[page-two]].
"""
    page1.write_text(page1_content, encoding="utf-8")
    
    page2 = wiki_dir / "page-two.md"
    page2_content = """---
title: Page Two
created: "2026-06-10T12:00:00Z"
updated: "2026-06-10T12:00:00Z"
sources: []
tags: []
---
Link to [[page-one]].
"""
    page2.write_text(page2_content, encoding="utf-8")
    
    violations = validate_vault(str(tmp_path))
    assert len(violations) == 0

def test_validate_vault_unresolved_link(tmp_path):
    # Setup vault skeleton
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    
    page1 = wiki_dir / "page-one.md"
    page1_content = """---
title: Page One
created: "2026-06-10T12:00:00Z"
updated: "2026-06-10T12:00:00Z"
sources: []
tags: []
---
Link to [[page-missing]].
"""
    page1.write_text(page1_content, encoding="utf-8")
    
    violations = validate_vault(str(tmp_path))
    assert len(violations) == 1
    assert violations[0].rule == "unresolved_link"
    assert "Outgoing link [[page-missing]] does not resolve" in violations[0].message
