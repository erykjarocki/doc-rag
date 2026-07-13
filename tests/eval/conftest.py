import json
import os
import sys
from pathlib import Path

import fitz
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

import src.config as config
import src.ingest as ingest
import src.qdrant_store as qdrant_store

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.embeddings import embed
from src.ingest import process_book
from src.qdrant_store import ensure_collection

BOOK_NAME = "tiny_sample"
COLLECTION_NAME = "tiny_sample"

REPORT_PATH = Path(__file__).parent / "eval-report.json"


def pytest_sessionfinish(session, exitstatus):
    """Print terminal summary and write eval-report.json after all tests."""
    results = getattr(session, "eval_results", [])
    if not results:
        return

    # Compute metrics
    recalls = [item["recall_at_k"] for item in results]
    precisions = [item["precision_at_k"] for item in results]
    rrs = [item["reciprocal_rank"] for item in results]

    avg_recall = sum(recalls) / len(recalls)
    avg_precision = sum(precisions) / len(precisions)
    avg_mrr = sum(rrs) / len(rrs)

    # Terminal summary
    terminal = session.config.get_terminal_writer()
    terminal.write("\n")
    terminal.write("=" * 70 + "\n")
    terminal.write("EVAL RESULTS\n")
    terminal.write("=" * 70 + "\n\n")

    for item in results:
        terminal.write(f'Query: "{item["query"]}"\n')
        for frag in item["retrieved_fragments"]:
            relevant = frag["is_relevant"]
            mark = "  \u2713 RELEVANT" if relevant else ""
            terminal.write(
                f'  [{frag["rank"]}] score={frag["score"]:.2f}'
                f'  page={frag["start_page"]}{mark}\n'
            )
            for line in frag["text"].split("\n"):
                terminal.write(f"      {line}\n")
            terminal.write("\n")
        terminal.write("\n")

    terminal.write("-" * 70 + "\n")
    terminal.write(
        f"Recall@2: {avg_recall:.2f} | "
        f"Precision@2: {avg_precision:.2f} | "
        f"MRR: {avg_mrr:.2f}\n"
    )
    terminal.write("-" * 70 + "\n\n")

    # Write JSON report
    report = {
        "queries": results,
        "metrics": {
            "recall_at_2": round(avg_recall, 4),
            "precision_at_2": round(avg_precision, 4),
            "mrr": round(avg_mrr, 4),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Generate detailed HTML report
    try:
        from tests.eval.generate_report import generate

        generate()
    except Exception:
        pass


def collect_eval_result(session, query, results, relevant_pages, k=2):
    """Run metrics on a query result and store on session for the summary hook."""
    if not hasattr(session, "eval_results"):
        session.eval_results = []

    # Deduplicate: only store once per query
    if any(item["query"] == query for item in session.eval_results):
        top_k = results[:k]
        return (
            _recall_at_k(results, relevant_pages, k),
            _precision_at_k(results, relevant_pages, k),
            _mrr(results, relevant_pages),
        )

    top_k = results[:k]
    rr = _mrr(results, relevant_pages)
    recall = _recall_at_k(results, relevant_pages, k)
    precision = _precision_at_k(results, relevant_pages, k)

    fragments = []
    for i, r in enumerate(top_k, 1):
        fragments.append(
            {
                "rank": i,
                "text": r["text"],
                "score": round(r["score"], 4),
                "start_page": r["start_page"],
                "end_page": r["end_page"],
                "chapter": r.get("chapter", ""),
                "is_relevant": r["start_page"] in relevant_pages,
            }
        )

    session.eval_results.append(
        {
            "query": query,
            "relevant_pages": relevant_pages,
            "retrieved_fragments": fragments,
            "recall_at_k": recall,
            "precision_at_k": precision,
            "reciprocal_rank": rr,
        }
    )

    return recall, precision, rr


def _precision_at_k(results, relevant_pages, k):
    top_k = results[:k]
    relevant = sum(1 for r in top_k if r["start_page"] in relevant_pages)
    return relevant / k


def _recall_at_k(results, relevant_pages, k):
    if not relevant_pages:
        return 1.0
    top_k = results[:k]
    found_pages = set(r["start_page"] for r in top_k if r["start_page"] in relevant_pages)
    return len(found_pages) / len(relevant_pages)


def _mrr(results, relevant_pages):
    for i, r in enumerate(results, 1):
        if r["start_page"] in relevant_pages:
            return 1.0 / i
    return 0.0


@pytest.fixture(scope="module")
def tiny_pdf(tmp_path_factory):
    """Generate a 3-page PDF about France, Germany, and Japan.

    Each page has ~1500 chars (~350 tokens) of distinct topic content,
    enough to produce separate chunks with the 384-token chunk size.
    """
    tmp_dir = tmp_path_factory.mktemp("pdfs")
    pdf_path = str(tmp_dir / "tiny_sample.pdf")

    texts = [
        (
            "Chapter 1: France\n\n"
            "Paris is the capital and most populous city of France, a global "
            "center for art, fashion, and gastronomy. The Eiffel Tower, "
            "constructed in 1889 for the World Fair, stands 330 meters tall "
            "and remains the most-visited paid monument in the world, drawing "
            "nearly seven million tourists annually. The tower was originally "
            "intended as a temporary structure and was nearly demolished after "
            "the exhibition ended.\n\n"
            "The Louvre Museum, located on the Right Bank of the Seine, is the "
            "largest art museum on Earth with over 380,000 objects in its "
            "permanent collection. Its iconic glass pyramid entrance, designed "
            "by architect I.M. Pei, was completed in 1989 and initially "
            "sparked controversy for its modern style against the classical "
            "palace. The museum houses the Mona Lisa by Leonardo da Vinci and "
            "the ancient Greek Venus de Milo sculpture, attracting more than "
            "nine million visitors each year.\n\n"
            "Notre-Dame de Paris is a medieval Catholic cathedral on the Ile "
            "de la Cite in the fourth arrondissement. Construction began in "
            "1163 under Bishop Maurice de Sully and was largely completed by "
            "1260. The cathedral is celebrated for its pioneering use of rib "
            "vaults and flying buttresses, elements that defined French Gothic "
            "architecture for centuries. A devastating fire in April 2019 "
            "destroyed the spire and much of the roof, prompting a massive "
            "international restoration effort."
        ),
        (
            "Chapter 2: Germany\n\n"
            "Berlin is the capital and largest city of Germany with "
            "approximately 3.7 million residents. The city is renowned for "
            "its vibrant arts scene, world-class museums, and complex history "
            "spanning from the Prussian empire through two world wars to Cold "
            "War division and eventual reunification. The Brandenburg Gate, "
            "constructed between 1788 and 1791 by architect Carl Gotthard "
            "Langhans, was originally a city gate and later became a symbol of "
            "both division and unity during the twentieth century.\n\n"
            "The Berlin Wall divided the city from August 13, 1961, until "
            "November 9, 1989. During those twenty-eight years, the concrete "
            "barrier separated families, friends, and an entire nation. "
            "Historical records indicate that at least 140 people died "
            "attempting to cross the wall. The fall of the wall was "
            "precipitated by a botched press conference by East German "
            "official Guenter Schabowski, who announced immediate border "
            "openings without consulting his superiors. Thousands gathered at "
            "the wall that night in celebrations marking the beginning of "
            "German reunification, officially completed on October 3, 1990.\n\n"
            "Munich, the capital of Bavaria, hosts Oktoberfest each year, the "
            "worlds largest folk festival running for sixteen days from "
            "mid-September. The celebration features traditional Bavarian "
            "music, food, and beer served in massive decorated tents, "
            "attracting over six million visitors annually. The festival "
            "traces its origins to 1810, when Crown Prince Ludwig of Bavaria "
            "married Princess Therese of Saxe-Hildburghausen."
        ),
        (
            "Chapter 3: Japan\n\n"
            "Tokyo is the capital and most populous metropolitan area of Japan, "
            "home to over 37 million people in the greater urban region. The "
            "city seamlessly blends ultramodern technology with ancient "
            "traditions, where centuries-old Shinto shrines stand alongside "
            "gleaming skyscrapers. Shibuya Crossing, the worlds busiest "
            "pedestrian intersection, handles up to 3,000 people per crossing "
            "cycle during peak hours and has become an iconic image of "
            "Japanese urban culture.\n\n"
            "Mount Fuji, standing at 3,776 meters above sea level, is Japans "
            "tallest peak and an active stratovolcano that last erupted in "
            "1707 during the Hoei eruption. The perfectly symmetrical volcanic "
            "cone is visible from Tokyo on clear winter days and has been a "
            "subject of Japanese art for centuries, most famously depicted in "
            "Katsushika Hokusais woodblock print series Thirty-six Views of "
            "Mount Fuji. UNESCO designated it as a World Heritage Site in "
            "2013.\n\n"
            "Japans Shinkansen bullet train network, operational since October "
            "1, 1964, connects major cities at speeds reaching 320 kilometers "
            "per hour. The Tokaido Shinkansen line between Tokyo and Osaka is "
            "the worlds busiest, carrying over 150 million passengers "
            "annually with an average delay of less than one minute. The "
            "system maintains a perfect safety record with zero passenger "
            "fatalities. Cherry blossoms, known as sakura, bloom across Japan "
            "each spring between late March and early April, drawing millions "
            "to hanami flower-viewing picnics beneath the blossoming trees."
        ),
    ]

    doc = fitz.open()
    for text in texts:
        page = doc.new_page()
        rect = page.rect
        page.insert_textbox(
            fitz.Rect(72, 72, rect.width - 72, rect.height - 72),
            text,
            fontsize=10,
        )

    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture(scope="module")
def indexed_qdrant(tiny_pdf, tmp_path_factory):
    """Run the full pipeline: extract -> chunk -> detect -> embed -> store.

    Uses real embedding model, real extraction, in-memory Qdrant.
    Returns (qdrant_client, collection_name, chunks).
    """
    tmp_dir = tmp_path_factory.mktemp("data")

    in_memory_client = QdrantClient(":memory:")
    original_client = qdrant_store._client

    qdrant_store._client = in_memory_client
    original_extracted = config.EXTRACTED_DIR
    config.EXTRACTED_DIR = str(tmp_dir / "extracted")
    ingest.EXTRACTED_DIR = config.EXTRACTED_DIR

    try:
        ensure_collection(COLLECTION_NAME, in_memory_client)

        result = process_book(tiny_pdf)
        chunks = result["chunks"]
        assert len(chunks) > 0, "Expected at least one chunk from the test PDF"

        texts = [c["text"] for c in chunks]
        vectors = embed(texts)

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(
                PointStruct(
                    id=i + 1,
                    vector=vector,
                    payload={
                        "text": chunk["text"],
                        "book": chunk["book"],
                        "chapter": chunk["chapter"],
                        "start_page": chunk["start_page"],
                        "end_page": chunk["end_page"],
                    },
                )
            )

        in_memory_client.upsert(collection_name=COLLECTION_NAME, points=points)
    except Exception:
        qdrant_store._client = original_client
        config.EXTRACTED_DIR = original_extracted
        ingest.EXTRACTED_DIR = original_extracted
        raise

    yield in_memory_client, COLLECTION_NAME, chunks

    qdrant_store._client = original_client
    config.EXTRACTED_DIR = original_extracted
    ingest.EXTRACTED_DIR = original_extracted
    try:
        in_memory_client.delete_collection(collection_name=COLLECTION_NAME)
    except Exception:
        pass
