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

from src.chunking import chunk_text
from src.embeddings import embed
from src.ingest import process_book
from src.qdrant_store import ensure_collection

BOOK_NAME = "tiny_sample"
COLLECTION_NAME = "tiny_sample"
GUTENBERG_COLLECTION = "gutenberg_prince"

REPORT_PATH = Path(__file__).parent / "eval-report.json"
BASELINE_PATH = Path(__file__).parent / "eval-baseline.json"


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

    # Show baseline delta if available
    baseline = None
    if BASELINE_PATH.exists():
        try:
            raw = json.loads(BASELINE_PATH.read_text())
            if "metrics" in raw:
                baseline = raw["metrics"]
            elif "recall_at_2" in raw:
                baseline = raw
        except (json.JSONDecodeError, KeyError):
            pass

    if baseline:
        def _fmt_delta(cur, base):
            diff = cur - base
            if abs(diff) < 0.005:
                return "= 0.00"
            sign = "+" if diff > 0 else ""
            return f"{sign}{diff:.2f}"

        terminal.write(
            f"          "
            f"(base: {_fmt_delta(avg_recall, baseline.get('recall_at_2', 0))} | "
            f"{_fmt_delta(avg_precision, baseline.get('precision_at_2', 0))} | "
            f"{_fmt_delta(avg_mrr, baseline.get('mrr', 0))})\n"
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

    Each page has ~3200 chars (~750 tokens) of distinct topic content.
    This ensures chunks stay within a single page (CHUNK_SIZE=384,
    CHUNK_OVERLAP=50), avoiding cross-page contamination.
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
            "the exhibition ended. Gustave Eiffel, the engineer who designed "
            "the tower, originally had a permit for it to stand for only "
            "twenty years. The tower has three levels for visitors, with "
            "restaurants on the first and second levels, and the top "
            "observation deck offering a panorama of Paris stretching nearly "
            "seventy kilometers in every direction on a clear day. The tower "
            "uses twenty thousand light bulbs to create a sparkling display "
            "every evening, a tradition that began in 1985 and has become "
            "one of the most recognizable features of the Parisian skyline.\n\n"
            "The Louvre Museum, located on the Right Bank of the Seine, is "
            "the largest art museum on Earth with over 380,000 objects in its "
            "permanent collection. Its iconic glass pyramid entrance, designed "
            "by architect I.M. Pei, was completed in 1989 and initially "
            "sparked controversy for its modern style against the classical "
            "palace. The museum houses the Mona Lisa by Leonardo da Vinci and "
            "the ancient Greek Venus de Milo sculpture, attracting more than "
            "nine million visitors each year. The collection spans from ancient "
            "Egyptian artifacts to paintings by Italian masters, and the "
            "museum itself occupies the former royal palace of the Kings of "
            "France, a building whose construction began in 1546 under King "
            "Francis I. The Louvre is so vast that a visitor spending thirty "
            "seconds at each artwork would need over one hundred days to see "
            "the entire collection.\n\n"
            "Notre-Dame de Paris is a medieval Catholic cathedral on the Ile "
            "de la Cite in the fourth arrondissement. Construction began in "
            "1163 under Bishop Maurice de Sully and was largely completed by "
            "1260. The cathedral is celebrated for its pioneering use of rib "
            "vaults and flying buttresses, elements that defined French Gothic "
            "architecture for centuries. A devastating fire in April 2019 "
            "destroyed the spire and much of the roof, prompting a massive "
            "international restoration effort. The cathedral reopened its "
            "doors in December 2024, five and a half years after the fire, "
            "with a ceremony attended by world leaders and religious figures. "
            "The restoration cost over eight hundred million euros and "
            "involved hundreds of craftsmen and artisans working to rebuild "
            "the structure using traditional methods and materials. The "
            "cathedral welcomes approximately thirteen million visitors each "
            "year, making it the most visited monument in all of France.\n\n"
            "The Palace of Versailles, located just twenty kilometers southwest "
            "of Paris, was the principal royal residence of France from 1682 "
            "until the start of the French Revolution in 1789. King Louis XIV, "
            "known as the Sun King, transformed a former hunting lodge into "
            "the opulent palace that became the envy of European courts. The "
            "Hall of Mirrors, the palace's most celebrated room, features 357 "
            "mirrors reflecting light from an equal number of arched windows, "
            "and served as the site where the Treaty of Versailles was signed "
            "in 1919, officially ending the First World War."
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
            "both division and unity during the twentieth century. The gate "
            "features twelve Doric columns supporting a bronze sculpture of "
            "a chariot drawn by four horses, known as the Quadriga, which "
            "was originally taken to Paris by Napoleon in 1806 and returned "
            "to Berlin after his defeat in 1814. The gate stands at the "
            "western end of Unter den Linden, a boulevard that once led "
            "directly to the royal palace of the Prussian monarchs.\n\n"
            "The Berlin Wall divided the city from August 13, 1961, until "
            "November 9, 1989. During those twenty-eight years, the concrete "
            "barrier separated families, friends, and an entire nation. "
            "Historical records indicate that at least 140 people died "
            "attempting to cross the wall. The fall of the wall was "
            "precipitated by a botched press conference by East German "
            "official Guenter Schabowski, who announced immediate border "
            "openings without consulting his superiors. Thousands gathered at "
            "the wall that night in celebrations marking the beginning of "
            "German reunification, officially completed on October 3, 1990. "
            "Today, a section of the wall remains standing near the "
            "Checkpoint Charlie crossing point as a memorial, and the "
            "surrounding area has been transformed into an open-air museum "
            "telling the story of the wall and the people affected by it. "
            "The East Side Gallery, a 1.3 kilometer stretch of the wall "
            "covered in murals by artists from around the world, is the "
            "longest surviving section and one of Berlin's most visited "
            "attractions.\n\n"
            "Munich, the capital of Bavaria, hosts Oktoberfest each year, the "
            "worlds largest folk festival running for sixteen days from "
            "mid-September. The celebration features traditional Bavarian "
            "music, food, and beer served in massive decorated tents, "
            "attracting over six million visitors annually. The festival "
            "traces its origins to 1810, when Crown Prince Ludwig of Bavaria "
            "married Princess Therese of Saxe-Hildburghausen. The "
            "Oktoberfest grounds, known as the Theresienwiese, cover "
            "forty-two hectares and include eighteen large beer tents, each "
            "able to seat several thousand guests. The festival has been "
            "cancelled only twenty-four times in its history, mostly due to "
            "wars and epidemics, and continues to be a major cultural "
            "attraction drawing visitors from around the world. Each tent "
            "is decorated in traditional Bavarian style and features its "
            "own brewery, with the Hofbraeuhaus tent being the largest "
             "and most famous, seating over ten thousand people.\n\n"
             "Cologne Cathedral, a Roman Catholic church on the banks of the "
             "Rhine River, took over six hundred years to complete. "
             "Construction began in 1248 and was not finished until 1880, "
             "making it one of the longest construction projects in history. "
             "The cathedral houses the Shrine of the Three Kings, a gold "
             "reliquary said to contain the remains of the Biblical Magi, "
             "and its twin spires rising 157 meters made it the tallest "
             "twin-spired building in the world upon completion. During World "
             "War Two, the cathedral survived fourteen aerial hits but "
             "remained standing amidst the rubble of the surrounding city."
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
            "Japanese urban culture. The district of Akihabara, once known "
            "for its electronics markets, has transformed into a center for "
            "anime, manga, and gaming culture, attracting enthusiasts from "
            "around the world to its countless shops and themed cafes. The "
            "Tsukiji Outer Market, though the inner wholesale market moved "
            "to Toyosu in 2018, remains a popular destination for fresh "
            "seafood and traditional Japanese street food.\n\n"
            "Mount Fuji, standing at 3,776 meters above sea level, is Japans "
            "tallest peak and an active stratovolcano that last erupted in "
            "1707 during the Hoei eruption. The perfectly symmetrical volcanic "
            "cone is visible from Tokyo on clear winter days and has been a "
            "subject of Japanese art for centuries, most famously depicted in "
            "Katsushika Hokusais woodblock print series Thirty-six Views of "
            "Mount Fuji. UNESCO designated it as a World Heritage Site in "
            "2013. The mountain is surrounded by five lakes, known as the "
            "Fuji Five Lakes, which are popular recreational areas offering "
            "hot springs, hiking trails, and boat tours with views of the "
            "iconic peak reflected in the calm waters. The climbing season "
            "runs from early July to early September, when mountain huts "
            "along the trails are open and the weather is most favorable "
            "for reaching the summit.\n\n"
            "Japans Shinkansen bullet train network, operational since October "
            "1, 1964, connects major cities at speeds reaching 320 kilometers "
            "per hour. The Tokaido Shinkansen line between Tokyo and Osaka is "
            "the worlds busiest, carrying over 150 million passengers "
            "annually with an average delay of less than one minute. The "
            "system maintains a perfect safety record with zero passenger "
            "fatalities. Cherry blossoms, known as sakura, bloom across Japan "
            "each spring between late March and early April, drawing millions "
            "to hanami flower-viewing picnics beneath the blossoming trees. "
            "The Japan Meteorological Agency issues annual blossom forecasts, "
            "tracking the sakura front as it moves northward from the "
            "southern island of Kyushu to the northern region of Hokkaido "
            "over a period of approximately two months. The blooming period "
            "lasts only about two weeks at each location, making timing "
            "essential for those hoping to experience the full beauty of "
            "the blossoms.\n\n"
            "Kyoto, the former imperial capital of Japan for over a thousand "
            "years, is home to seventeen UNESCO World Heritage Sites and more "
            "than two thousand temples and shrines. The Fushimi Inari Shrine, "
            "famous for its thousands of vermilion torii gates forming winding "
            "paths up a mountainside, attracts millions of visitors each year "
            "and is one of the most photographed locations in all of Japan. "
            "The Kinkaku-ji Golden Pavilion, covered in gold leaf and set "
            "against a backdrop of maple trees, was originally built in 1397 "
            "as a retirement villa for the Shogun Ashikaga Yoshimitsu."
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


# ---------------------------------------------------------------------------
# Gutenberg corpus fixtures (real content, no PDF)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gutenberg_corpus():
    """Fetch The Prince from Gutenberg, split into chapters.

    Returns (text, page_boundaries, page_nums) — 26 chapters.
    Cached via lru_cache so multiple fixtures share one fetch.
    """
    from tests.eval.gutenberg_corpus import fetch_and_split
    return fetch_and_split()


@pytest.fixture(scope="module")
def gutenberg_indexed_qdrant(gutenberg_corpus, tmp_path_factory):
    """Chunk, embed, and index the Gutenberg corpus into in-memory Qdrant.

    Uses chunk_text() directly — no PDF parsing, no extraction.
    Returns (qdrant_client, collection_name, chunks).
    """
    text, page_boundaries, page_nums = gutenberg_corpus
    tmp_dir = tmp_path_factory.mktemp("gutenberg")

    in_memory_client = QdrantClient(":memory:")
    original_client = qdrant_store._client

    qdrant_store._client = in_memory_client
    original_extracted = config.EXTRACTED_DIR
    config.EXTRACTED_DIR = str(tmp_dir / "extracted")
    ingest.EXTRACTED_DIR = config.EXTRACTED_DIR

    try:
        ensure_collection(GUTENBERG_COLLECTION, in_memory_client)

        chunks = chunk_text(text, page_boundaries, page_nums)
        assert len(chunks) > 0, "Expected at least one chunk from Gutenberg corpus"

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
                        "book": "gutenberg_prince",
                        "chapter": f"Chapter {chunk['start_page']}",
                        "start_page": chunk["start_page"],
                        "end_page": chunk["end_page"],
                    },
                )
            )

        in_memory_client.upsert(collection_name=GUTENBERG_COLLECTION, points=points)
    except Exception:
        qdrant_store._client = original_client
        config.EXTRACTED_DIR = original_extracted
        ingest.EXTRACTED_DIR = original_extracted
        raise

    yield in_memory_client, GUTENBERG_COLLECTION, chunks

    qdrant_store._client = original_client
    config.EXTRACTED_DIR = original_extracted
    ingest.EXTRACTED_DIR = original_extracted
    try:
        in_memory_client.delete_collection(collection_name=GUTENBERG_COLLECTION)
    except Exception:
        pass
