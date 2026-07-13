import pytest

from src.extraction import (
    get_full_text_with_page_info,
    get_page_boundaries,
    page_at_position,
)


@pytest.mark.unit
class TestPageBoundaries:
    def test_single_page(self, sample_pages):
        boundaries = get_page_boundaries([sample_pages[0]])
        assert len(boundaries) == 1
        assert boundaries[0] == len(sample_pages[0]["text"]) + 1

    def test_multiple_pages(self, sample_pages):
        boundaries = get_page_boundaries(sample_pages)
        assert len(boundaries) == 3
        assert boundaries[0] < boundaries[1] < boundaries[2]


@pytest.mark.unit
class TestGetFullTextWithPageInfo:
    def test_full_text_contains_all_pages(self, sample_pages):
        full_text, _ = get_full_text_with_page_info(sample_pages)
        for page in sample_pages:
            assert page["text"] in full_text

    def test_boundaries_length_matches_pages(self, sample_pages):
        _, boundaries = get_full_text_with_page_info(sample_pages)
        assert len(boundaries) == len(sample_pages)


@pytest.mark.unit
class TestPageAtPosition:
    def test_first_page(self, sample_pages):
        boundaries = get_page_boundaries(sample_pages)
        page_nums = [p["page_num"] for p in sample_pages]
        assert page_at_position(boundaries, page_nums, 0) == 1

    def test_last_page(self, sample_pages):
        boundaries = get_page_boundaries(sample_pages)
        page_nums = [p["page_num"] for p in sample_pages]
        last_pos = boundaries[-1] - 1
        assert page_at_position(boundaries, page_nums, last_pos) == 3

    def test_beyond_last_page(self, sample_pages):
        boundaries = get_page_boundaries(sample_pages)
        page_nums = [p["page_num"] for p in sample_pages]
        assert page_at_position(boundaries, page_nums, 999999) == 3
