from pathlib import Path
import re
import textwrap
from typing import Any


MARKDOWN_HEADING_RE = re.compile(r"^(#{1,2})\s+(.+?)\s*#*\s*$")
RST_CODE_DIRECTIVE_RE = re.compile(r"^(\s*)\.\.\s+(code|code-block)::\s*(\S+)?\s*$")


def extract_document_section_records_from_file(
    document_path: str | Path,
) -> list[dict[str, Any]]:
    """Split one text file into document-section records."""

    document_path = Path(document_path)
    document_text = document_path.read_text(encoding="utf-8")
    return extract_document_section_records_from_text(document_text, document_path)


def extract_document_section_records_from_text(
    document_text: str,
    document_path: str | Path,
) -> list[dict[str, Any]]:
    """Split document text by title/section headings."""

    document_path = Path(document_path)
    lines = document_text.splitlines()
    headings = extract_document_section_headings(document_text)
    document_title = extract_document_title(headings, document_path)

    if not headings:
        headings = [{"heading": document_title, "start": 0}]

    records = []
    for index, heading in enumerate(headings):
        next_start = headings[index + 1]["start"] if index + 1 < len(headings) else len(lines)
        section_text = "\n".join(lines[heading["start"] : next_start]).strip()
        records.append(
            build_document_section_record(
                document_path=document_path,
                document_title=document_title,
                document_section=heading["heading"],
                document_section_text=section_text,
                document_section_index=index + 1,
                document_section_count=len(headings),
            )
        )

    return records


def extract_document_title(
    headings: list[dict[str, Any]],
    document_path: str | Path,
) -> str:
    if headings:
        return headings[0]["heading"]
    return Path(document_path).stem.replace("-", " ").replace("_", " ").title()


def extract_document_section_headings(document_text: str) -> list[dict[str, Any]]:
    """Find Markdown #/## and reStructuredText =/- headings."""

    headings = []
    lines = document_text.splitlines()
    in_fenced_code = False
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            in_fenced_code = not in_fenced_code
            index += 1
            continue

        if in_fenced_code:
            index += 1
            continue

        markdown_match = MARKDOWN_HEADING_RE.match(stripped)
        if markdown_match:
            headings.append(
                {
                    "heading": markdown_match.group(2).strip(),
                    "level": len(markdown_match.group(1)),
                    "style": "markdown",
                    "start": index,
                }
            )
            index += 1
            continue

        if index + 1 < len(lines) and is_rst_heading(line, lines[index + 1]):
            underline = lines[index + 1].strip()
            headings.append(
                {
                    "heading": stripped,
                    "level": 1 if underline[0] == "=" else 2,
                    "style": "rst",
                    "start": index,
                }
            )
            index += 2
            continue

        index += 1

    return headings


def is_rst_heading(title_line: str, underline_line: str) -> bool:
    title = title_line.strip()
    underline = underline_line.strip()

    return (
        bool(title)
        and len(underline) >= max(3, len(title))
        and underline[0] in {"=", "-"}
        and set(underline) == {underline[0]}
        and not title_line.lstrip().startswith((".. ", ">>> ", "... "))
    )


def extract_code_blocks(text: str) -> list[str]:
    """Extract Markdown fenced code and simple reStructuredText code blocks."""

    code_blocks = []
    lines = text.splitlines()
    index = 0

    while index < len(lines):
        stripped = lines[index].strip()

        if stripped.startswith("```"):
            code, index = consume_fenced_code(lines, index)
            code_blocks.append(code)
            continue

        directive_match = RST_CODE_DIRECTIVE_RE.match(lines[index])
        if directive_match:
            code, index = consume_rst_code_block(
                lines,
                start_index=index,
                directive_indent=len(directive_match.group(1)),
            )
            code_blocks.append(code)
            continue

        index += 1

    return [code for code in code_blocks if code]


def consume_fenced_code(lines: list[str], start_index: int) -> tuple[str, int]:
    code_lines = []
    index = start_index + 1

    while index < len(lines) and not lines[index].strip().startswith("```"):
        code_lines.append(lines[index])
        index += 1

    return "\n".join(code_lines).strip("\n"), index + 1


def consume_rst_code_block(
    lines: list[str],
    start_index: int,
    directive_indent: int,
) -> tuple[str, int]:
    index = start_index + 1

    while index < len(lines) and (
        lines[index].strip() == ""
        or lines[index].strip().startswith(":")
    ):
        index += 1

    code_lines = []
    while index < len(lines):
        line = lines[index]
        if line.strip() and indent_width(line) <= directive_indent:
            break
        code_lines.append(line)
        index += 1

    code = textwrap.dedent("\n".join(code_lines)).strip("\n")
    return code, index


def build_document_section_record(
    document_path: Path,
    document_title: str,
    document_section: str,
    document_section_text: str,
    document_section_index: int,
    document_section_count: int,
) -> dict[str, Any]:
    code_blocks = extract_code_blocks(document_section_text)
    return {
        "text": document_section_text,
        "metadata": {
            "document_filename": document_path.name,
            "document_filepath": str(document_path),
            "document_title": document_title,
            "document_section": document_section,
            "document_section_index": document_section_index,
            "document_section_count": document_section_count,
            "has_code": bool(code_blocks),
            "code": code_blocks,
        },
    }


def indent_width(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))
