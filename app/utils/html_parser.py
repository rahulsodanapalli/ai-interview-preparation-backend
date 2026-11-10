"""Zero-dependency HTML text and title extractor using Python's standard library."""

from html.parser import HTMLParser


class HTMLTextExtractor(HTMLParser):
    """Safe, zero-dependency HTML text and title parser using Python's standard library html.parser."""

    SKIPPED_TAGS = {
        "script",
        "style",
        "noscript",
        "svg",
        "nav",
        "footer",
        "header",
        "aside",
        "form",
        "button",
        "input",
        "select",
        "textarea",
    }

    BLOCK_TAGS = {
        "p",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "ul",
        "ol",
        "tr",
        "td",
        "th",
        "article",
        "section",
        "blockquote",
        "pre",
        "hr",
        "br",
    }

    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []
        self.title_parts: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        tag_lower = tag.lower()
        if tag_lower in self.SKIPPED_TAGS:
            self._skip_depth += 1
            return

        if tag_lower == "title":
            self._in_title = True
        elif tag_lower in self.BLOCK_TAGS and self._skip_depth == 0:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in self.SKIPPED_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return

        if tag_lower == "title":
            self._in_title = False
        elif tag_lower in self.BLOCK_TAGS and self._skip_depth == 0:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        elif self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self.text_parts.append(f" {stripped} ")

    def get_extracted_title(self) -> str:
        raw = "".join(self.title_parts).strip()
        return " ".join(raw.split()) if raw else "Webpage Documentation"

    def get_extracted_text(self) -> str:
        raw = "".join(self.text_parts)
        lines = [line.strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)

