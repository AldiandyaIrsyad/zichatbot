import pytest

from app.rag.chunking.fixed import (
    create_parent_chunks_fixed,
    split_into_children_fixed,
)
from app.rag.chunking.logic import MIN_CHILD_TEXT_LENGTH
from app.rag.chunking.models import ContentType, ParentChunkData, ParsedElement


class WordTokenizer:
    """Whitespace tokenizer standing in for BGE-M3's.

    Keeps the tests free of `transformers` (and of a model download) while
    exercising the same encode/decode contract: one token per word, decoding
    back to space-joined text.
    """

    def __init__(self):
        self._vocab: list[str] = []
        self._ids: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self._ids:
                self._ids[word] = len(self._vocab)
                self._vocab.append(word)
            ids.append(self._ids[word])
        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(self._vocab[i] for i in token_ids)


@pytest.fixture
def tok():
    return WordTokenizer()


def _element(text: str, element_type: str = "NarrativeText", page: int | None = 1):
    return ParsedElement(
        element_type=element_type, text=text, metadata={"page_number": page}
    )


def _words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


# ---------------------------------------------------------------------------
# create_parent_chunks_fixed
# ---------------------------------------------------------------------------

class TestCreateParentChunksFixed:

    def test_empty_elements_returns_empty(self, tok):
        assert create_parent_chunks_fixed([], "doc-1", tok) == []

    def test_elements_with_only_ignored_types_returns_empty(self, tok):
        elements = [
            _element("page 1", element_type="Header"),
            _element("footer text", element_type="Footer"),
            _element("12", element_type="PageNumber"),
        ]
        assert create_parent_chunks_fixed(elements, "doc-1", tok) == []

    def test_windows_respect_max_tokens(self, tok):
        elements = [_element(_words(25))]
        parents = create_parent_chunks_fixed(elements, "doc-1", tok, max_tokens=10)

        assert len(parents) == 3
        assert [len(tok.encode(p.text)) for p in parents] == [10, 10, 5]

    def test_parent_windows_do_not_overlap(self, tok):
        elements = [_element(_words(20))]
        parents = create_parent_chunks_fixed(elements, "doc-1", tok, max_tokens=10)

        rejoined = " ".join(p.text for p in parents)
        assert rejoined == _words(20)

    def test_flat_hierarchy_fields(self, tok):
        elements = [_element(_words(15))]
        parents = create_parent_chunks_fixed(elements, "doc-1", tok, max_tokens=10)

        for i, parent in enumerate(parents):
            assert parent.breadcrumbs == []
            assert parent.depth == 0
            assert parent.parent_id is None
            assert parent.path == "doc-1"
            assert parent.chunk_index == i
            assert parent.ordinal == i
            assert parent.content_type == ContentType.TEXT
            assert parent.doc_id == "doc-1"

    def test_headings_do_not_create_boundaries(self, tok):
        """Unlike the hierarchical chunker, a Title is just more tokens."""
        elements = [
            _element("BAB I", element_type="Title"),
            _element(_words(3)),
            _element("Pasal 1", element_type="Title"),
            _element(_words(3)),
        ]
        parents = create_parent_chunks_fixed(elements, "doc-1", tok, max_tokens=100)

        assert len(parents) == 1
        assert parents[0].breadcrumbs == []
        assert "BAB I" in parents[0].text and "Pasal 1" in parents[0].text

    def test_page_attribution_follows_window_start(self, tok):
        elements = [
            _element(_words(10, "a"), page=3),
            _element(_words(10, "b"), page=7),
        ]
        parents = create_parent_chunks_fixed(elements, "doc-1", tok, max_tokens=10)

        assert [p.page for p in parents] == [3, 7]

    def test_ignored_elements_are_skipped(self, tok):
        elements = [
            _element("Halaman 1", element_type="Header"),
            _element(_words(5, "body")),
            _element("2", element_type="PageNumber"),
        ]
        parents = create_parent_chunks_fixed(elements, "doc-1", tok, max_tokens=100)

        assert len(parents) == 1
        assert parents[0].text == _words(5, "body")

    def test_blank_elements_are_skipped(self, tok):
        elements = [_element("   "), _element(_words(3, "x"))]
        parents = create_parent_chunks_fixed(elements, "doc-1", tok, max_tokens=100)

        assert len(parents) == 1
        assert parents[0].text == _words(3, "x")

    def test_ids_are_unique(self, tok):
        elements = [_element(_words(50))]
        parents = create_parent_chunks_fixed(elements, "doc-1", tok, max_tokens=10)

        assert len({p.id for p in parents}) == len(parents)


# ---------------------------------------------------------------------------
# split_into_children_fixed
# ---------------------------------------------------------------------------

def _parent(text: str, **overrides) -> ParentChunkData:
    kwargs = dict(
        id="parent-1",
        doc_id="doc-1",
        text=text,
        chunk_index=0,
        page=4,
        breadcrumbs=[],
        content_type=ContentType.TEXT,
        path="doc-1",
        depth=0,
    )
    kwargs.update(overrides)
    return ParentChunkData(**kwargs)


class TestSplitIntoChildrenFixed:

    def test_windows_respect_max_tokens(self, tok):
        children = split_into_children_fixed(
            _parent(_words(25)), tok, max_tokens=10, overlap_tokens=2
        )
        assert all(len(tok.encode(c.text)) <= 10 for c in children)

    def test_overlap_is_applied(self, tok):
        children = split_into_children_fixed(
            _parent(_words(20)), tok, max_tokens=10, overlap_tokens=4
        )
        first = children[0].text.split()
        second = children[1].text.split()

        assert first[-4:] == second[:4]

    def test_step_covers_all_tokens(self, tok):
        children = split_into_children_fixed(
            _parent(_words(20)), tok, max_tokens=10, overlap_tokens=4
        )
        covered = {word for c in children for word in c.text.split()}
        assert covered == set(_words(20).split())

    def test_final_short_window_emitted_once(self, tok):
        children = split_into_children_fixed(
            _parent(_words(13)), tok, max_tokens=10, overlap_tokens=4
        )
        assert [c.text for c in children].count(children[-1].text) == 1

    def test_ordinals_and_paths_are_contiguous(self, tok):
        children = split_into_children_fixed(
            _parent(_words(30)), tok, max_tokens=10, overlap_tokens=2
        )
        assert [c.ordinal for c in children] == list(range(len(children)))
        assert [c.path for c in children] == [
            f"doc-1.c{i}" for i in range(len(children))
        ]

    def test_short_children_are_dropped(self, tok):
        """A trailing window under MIN_CHILD_TEXT_LENGTH chars is gibberish."""
        assert MIN_CHILD_TEXT_LENGTH == 8
        children = split_into_children_fixed(
            _parent("aaaaaaaaaa bbbbbbbbbb x"), tok, max_tokens=2, overlap_tokens=0
        )
        assert [c.text for c in children] == ["aaaaaaaaaa bbbbbbbbbb"]

    def test_children_inherit_parent_metadata(self, tok):
        parent = _parent(_words(20))
        children = split_into_children_fixed(
            parent, tok, max_tokens=10, overlap_tokens=2
        )
        for child in children:
            assert child.parent_chunk_id == parent.id
            assert child.doc_id == parent.doc_id
            assert child.page == parent.page
            assert child.content_type == parent.content_type

    def test_no_breadcrumb_tag_prepended(self, tok):
        children = split_into_children_fixed(
            _parent(_words(10, "z")), tok, max_tokens=100, overlap_tokens=0
        )
        assert children[0].text == _words(10, "z")

    def test_overlap_not_smaller_than_max_raises(self, tok):
        with pytest.raises(ValueError):
            split_into_children_fixed(
                _parent(_words(10)), tok, max_tokens=8, overlap_tokens=8
            )

    def test_empty_parent_text_returns_empty(self, tok):
        assert split_into_children_fixed(_parent(""), tok) == []
