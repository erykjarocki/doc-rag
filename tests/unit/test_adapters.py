import pytest

from src.adapters import (
    CodeAdapter,
    DocumentSection,
    MarkdownAdapter,
    PlainTextAdapter,
    get_adapter,
    section_for_position,
    supported_extensions,
)


@pytest.mark.unit
class TestDocumentSection:
    def test_creation(self):
        sec = DocumentSection(text="hello", section_name="intro", start_line=1, end_line=5)
        assert sec.text == "hello"
        assert sec.section_name == "intro"
        assert sec.start_line == 1
        assert sec.end_line == 5


@pytest.mark.unit
class TestSectionForPosition:
    def test_single_section(self):
        sec = DocumentSection(
            text="hello world", section_name="all", start_line=1, end_line=1
        )
        assert section_for_position([sec], 0, "hello world") == "all"

    def test_multiple_sections(self):
        sections = [
            DocumentSection(text="intro", section_name="Introduction", start_line=1, end_line=3),
            DocumentSection(text="body", section_name="Body", start_line=4, end_line=6),
        ]
        # Position 0 should be in first section
        result = section_for_position(sections, 0, "intro\nmore intro\n\nbody\nmore body")
        assert result == "Introduction"

    def test_empty_sections(self):
        assert section_for_position([], 0, "text") == "unknown"


@pytest.mark.unit
class TestPlainTextAdapter:
    def test_extract(self, tmp_path):
        txt = tmp_path / "hello.txt"
        txt.write_text("Line 1\nLine 2\nLine 3\n", encoding="utf-8")

        adapter = PlainTextAdapter()
        doc = adapter.extract(str(txt))

        assert doc.name == "hello"
        assert doc.full_text == "Line 1\nLine 2\nLine 3\n"
        assert len(doc.sections) == 1
        assert doc.sections[0].section_name == "full_text"
        assert doc.sections[0].start_line == 1
        assert doc.sections[0].end_line == 4
        assert doc.page_nums == [1]
        assert len(doc.page_boundaries) == 1

    def test_empty_file(self, tmp_path):
        txt = tmp_path / "empty.txt"
        txt.write_text("", encoding="utf-8")

        adapter = PlainTextAdapter()
        doc = adapter.extract(str(txt))
        assert doc.name == "empty"
        assert doc.full_text == ""
        assert len(doc.sections) == 1

    def test_unicode(self, tmp_path):
        txt = tmp_path / "polski.txt"
        txt.write_text("Zażółć gęślą jaźń\nŁódź\n", encoding="utf-8")

        adapter = PlainTextAdapter()
        doc = adapter.extract(str(txt))
        assert "Zażółć" in doc.full_text
        assert doc.name == "polski"


@pytest.mark.unit
class TestMarkdownAdapter:
    def test_heading_sections(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text(
            "# Title\nSome intro\n\n## Chapter 1\nContent here\n\n## Chapter 2\nMore content\n",
            encoding="utf-8",
        )

        adapter = MarkdownAdapter()
        doc = adapter.extract(str(md))

        assert doc.name == "doc"
        assert len(doc.sections) >= 2  # At least heading sections
        names = [s.section_name for s in doc.sections]
        assert "Title" in names
        assert "Chapter 1" in names
        assert "Chapter 2" in names

    def test_no_headings(self, tmp_path):
        md = tmp_path / "plain.md"
        md.write_text("Just some text\nNo headings here\n", encoding="utf-8")

        adapter = MarkdownAdapter()
        doc = adapter.extract(str(md))
        assert len(doc.sections) == 1
        assert doc.sections[0].section_name == "full_text"

    def test_preamble_before_first_heading(self, tmp_path):
        md = tmp_path / "preamble.md"
        md.write_text("Intro text\n\n# First Heading\nBody\n", encoding="utf-8")

        adapter = MarkdownAdapter()
        doc = adapter.extract(str(md))
        names = [s.section_name for s in doc.sections]
        assert "preamble" in names
        assert "First Heading" in names

    def test_single_page_for_non_pdf(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("# Hello\nWorld\n", encoding="utf-8")

        adapter = MarkdownAdapter()
        doc = adapter.extract(str(md))
        assert doc.page_nums == [1]


@pytest.mark.unit
class TestCodeAdapter:
    def test_python_functions(self, tmp_path):
        py = tmp_path / "module.py"
        py.write_text(
            "import os\n\n\ndef hello():\n    pass\n\n\ndef world():\n    pass\n",
            encoding="utf-8",
        )

        adapter = CodeAdapter()
        doc = adapter.extract(str(py))

        names = [s.section_name for s in doc.sections]
        assert "hello" in names
        assert "world" in names

    def test_python_class(self, tmp_path):
        py = tmp_path / "cls.py"
        py.write_text(
            "class MyClass:\n    def method(self):\n        pass\n",
            encoding="utf-8",
        )

        adapter = CodeAdapter()
        doc = adapter.extract(str(py))
        names = [s.section_name for s in doc.sections]
        assert "MyClass" in names

    def test_javascript_functions(self, tmp_path):
        js = tmp_path / "app.js"
        js.write_text(
            "const x = 1;\n\nfunction greet() {\n  return 'hi';\n}\n\n"
            "function farewell() {\n  return 'bye';\n}\n",
            encoding="utf-8",
        )

        adapter = CodeAdapter()
        doc = adapter.extract(str(js))
        names = [s.section_name for s in doc.sections]
        assert "greet" in names
        assert "farewell" in names

    def test_no_sections(self, tmp_path):
        txt = tmp_path / "config.txt"
        txt.write_text("key = value\nfoo = bar\n", encoding="utf-8")

        adapter = CodeAdapter()
        doc = adapter.extract(str(txt))
        assert len(doc.sections) == 1
        assert doc.sections[0].section_name == "full_file"

    def test_rust_functions(self, tmp_path):
        rs = tmp_path / "main.rs"
        rs.write_text(
            "fn main() {\n    println!(\"hi\");\n}\n\nfn helper() -> i32 {\n    42\n}\n",
            encoding="utf-8",
        )

        adapter = CodeAdapter()
        doc = adapter.extract(str(rs))
        names = [s.section_name for s in doc.sections]
        assert "main" in names
        assert "helper" in names


@pytest.mark.unit
class TestGetAdapter:
    def test_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.touch()
        adapter = get_adapter(str(f))
        assert isinstance(adapter, PlainTextAdapter)

    def test_md(self, tmp_path):
        f = tmp_path / "test.md"
        f.touch()
        adapter = get_adapter(str(f))
        assert isinstance(adapter, MarkdownAdapter)

    def test_py(self, tmp_path):
        f = tmp_path / "test.py"
        f.touch()
        adapter = get_adapter(str(f))
        assert isinstance(adapter, CodeAdapter)

    def test_unsupported(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.touch()
        with pytest.raises(ValueError, match="Unsupported file format"):
            get_adapter(str(f))

    def test_case_insensitive(self, tmp_path):
        f = tmp_path / "TEST.PY"
        f.touch()
        adapter = get_adapter(str(f))
        assert isinstance(adapter, CodeAdapter)


@pytest.mark.unit
class TestSupportedExtensions:
    def test_includes_common_formats(self):
        exts = supported_extensions()
        assert ".pdf" in exts
        assert ".txt" in exts
        assert ".md" in exts
        assert ".py" in exts
        assert ".js" in exts

    def test_returns_sorted_unique(self):
        exts = supported_extensions()
        assert exts == sorted(set(exts))
