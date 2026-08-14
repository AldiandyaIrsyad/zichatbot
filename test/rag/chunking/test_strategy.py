import pytest

from app.rag.chunking.config import ChunkingConfig
from app.rag.chunking.logic import create_parent_chunks, split_into_children
from app.rag.chunking.models import ContentType, ParsedElement
from app.rag.chunking.strategy import chunk_children, chunk_parents

from test.rag.chunking.test_fixed import WordTokenizer


@pytest.fixture
def tok():
    return WordTokenizer()


@pytest.fixture
def elements():
    return [
        ParsedElement(element_type="Title", text="BAB I", metadata={"page_number": 1}),
        ParsedElement(
            element_type="NarrativeText",
            text="Ketentuan umum yang berlaku bagi seluruh sivitas akademika universitas.",
            metadata={"page_number": 1},
        ),
        ParsedElement(element_type="Title", text="Pasal 1", metadata={"page_number": 2}),
        ParsedElement(
            element_type="NarrativeText",
            text="Dalam peraturan ini yang dimaksud dengan rektor adalah pimpinan universitas.",
            metadata={"page_number": 2},
        ),
    ]


class TestConfigDefaults:

    def test_default_strategy_is_hierarchical(self):
        assert ChunkingConfig().strategy == "hierarchical"

    def test_overlap_must_be_smaller_than_window(self):
        with pytest.raises(ValueError):
            ChunkingConfig(fixed_child_max_tokens=512, fixed_child_overlap_tokens=512)


class TestHierarchicalDispatch:

    def test_default_config_matches_create_parent_chunks(self, elements):
        via_strategy = chunk_parents(elements, "doc-1", ChunkingConfig())
        direct = create_parent_chunks(elements, "doc-1")

        assert len(via_strategy) == len(direct)
        assert [p.text for p in via_strategy] == [p.text for p in direct]
        assert [p.breadcrumbs for p in via_strategy] == [p.breadcrumbs for p in direct]
        assert [p.path for p in via_strategy] == [p.path for p in direct]
        assert [p.depth for p in via_strategy] == [p.depth for p in direct]

    def test_default_config_matches_split_into_children(self, elements):
        parent = create_parent_chunks(elements, "doc-1")[0]

        via_strategy = chunk_children(parent, ChunkingConfig())
        direct = split_into_children(parent)

        assert [c.text for c in via_strategy] == [c.text for c in direct]
        assert [c.path for c in via_strategy] == [c.path for c in direct]

    def test_hierarchical_needs_no_tokenizer(self, elements):
        assert chunk_parents(elements, "doc-1", ChunkingConfig(), tokenizer=None)

    def test_sizes_are_honored(self, elements):
        config = ChunkingConfig(parent_max_chars=40)
        assert len(chunk_parents(elements, "doc-1", config)) > len(
            chunk_parents(elements, "doc-1", ChunkingConfig())
        )


class TestFixedDispatch:

    def test_routes_to_fixed_implementation(self, elements, tok):
        config = ChunkingConfig(strategy="fixed", fixed_parent_max_tokens=4)
        parents = chunk_parents(elements, "doc-1", config, tok)

        assert len(parents) > 1
        assert all(p.breadcrumbs == [] for p in parents)
        assert all(p.depth == 0 and p.path == "doc-1" for p in parents)
        assert all(p.content_type == ContentType.TEXT for p in parents)

    def test_children_use_token_windows(self, elements, tok):
        config = ChunkingConfig(
            strategy="fixed", fixed_child_max_tokens=4, fixed_child_overlap_tokens=1
        )
        parent = chunk_parents(elements, "doc-1", ChunkingConfig(), None)[0]
        children = chunk_children(parent, config, tok)

        assert children
        assert all(len(tok.encode(c.text)) <= 4 for c in children)

    def test_missing_tokenizer_raises(self, elements):
        config = ChunkingConfig(strategy="fixed")

        with pytest.raises(ValueError, match="requires a tokenizer"):
            chunk_parents(elements, "doc-1", config, tokenizer=None)

    def test_missing_tokenizer_raises_for_children(self, elements):
        config = ChunkingConfig(strategy="fixed")
        parent = chunk_parents(elements, "doc-1", ChunkingConfig(), None)[0]

        with pytest.raises(ValueError, match="requires a tokenizer"):
            chunk_children(parent, config, tokenizer=None)
