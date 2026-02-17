import json
import uuid
import os
from bs4 import BeautifulSoup
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, CompositeElement
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser


# ===============================
# 1. PDF Partition
# ===============================

def create_chunks_from_pdf(file_path):
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    return elements


# ===============================
# 2. Element Segregation
# ===============================

def table_text_segregation(all_elements):
    tables = []
    texts  = []
    for el in all_elements:
        if isinstance(el, Table):
            tables.append(el)
        elif isinstance(el, CompositeElement):
            texts.append(el)
    return tables, texts


def get_images(chunks):
    images_b64 = []
    for chunk in chunks:
        if isinstance(chunk, CompositeElement):
            chunk_els = chunk.metadata.orig_elements or []
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64


# ===============================
# 3. Text Normalization
# ===============================

def clean_text(text):
    """Remove extra whitespace. Keep the content intact."""
    return " ".join(text.split())


def html_table_to_text(html):
    """
    Convert HTML table to plain text rows.
    Each row: col1 | col2 | col3
    Keeps newlines between rows — do NOT call clean_text on this.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        cells = [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


# ===============================
# 4. Image Description via LLaVA
# ===============================

def describe_image(image_b64):
    """
    Use LLaVA (via Ollama) to describe an image.
    This is the only place we use a model during ingestion.
    LLaVA is necessary here — images have no raw text form.

    If Ollama is not running or LLaVA is not installed,
    returns a fallback string so ingestion does not crash.
    """
    try:
        llm    = ChatOllama(model="llava", temperature=0.0)
        parser = StrOutputParser()

        data_url = f"data:image/jpeg;base64,{image_b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in technical detail. "
                            "If it contains a chart, table, or diagram, "
                            "describe the values and structure precisely."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

        resp = llm.invoke(messages)
        return parser.invoke(resp)

    except Exception as e:
        print(f"   WARNING: LLaVA failed for image — {e}")
        return "Image description unavailable."


# ===============================
# 5. Export
# ===============================

def export_chunk(record, output_file):
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ===============================
# 6. Main Processing Loop
# ===============================

def process_pdfs_in_directory(directory_path, output_file="chunks.jsonl"):

    # Clear output file at start of each run — prevents stale data
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Cleared old {output_file}")

    total_texts  = 0
    total_tables = 0
    total_images = 0

    for filename in os.listdir(directory_path):
        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(directory_path, filename)
        elements  = create_chunks_from_pdf(file_path)

        print(f"\n📘 Processing: {filename}")
        print(f"   Elements found: {len(elements)}")

        tables, texts = table_text_segregation(elements)
        images        = get_images(elements)

        print(f"   Texts: {len(texts)}, Tables: {len(tables)}, Images: {len(images)}")

        # -------- TEXT CHUNKS --------
        # retrieval_text = clean raw text
        # No summarisation — BM25 needs real keywords, dense needs real content
        for text in texts:
            raw_text = text.text if hasattr(text, "text") else str(text)
            raw_text = clean_text(raw_text)

            record = {
                "chunk_id"      : str(uuid.uuid4()),
                "modality"      : "text",
                "source_pdf"    : filename,
                "page_number"   : getattr(text.metadata, "page_number", None),
                "raw_text"      : raw_text,
                "image_b64"     : None,
                "retrieval_text": raw_text,
            }
            export_chunk(record, output_file)

        # -------- TABLE CHUNKS --------
        # retrieval_text = structured plain text rows
        # HTML converted to "col1 | col2" format — readable by both BM25 and dense
        for table in tables:
            if hasattr(table, "metadata") and hasattr(table.metadata, "text_as_html"):
                table_text = html_table_to_text(table.metadata.text_as_html)
            else:
                table_text = clean_text(str(table))

            record = {
                "chunk_id"      : str(uuid.uuid4()),
                "modality"      : "table",
                "source_pdf"    : filename,
                "page_number"   : getattr(table.metadata, "page_number", None),
                "raw_text"      : table_text,
                "image_b64"     : None,
                "retrieval_text": table_text,
            }
            export_chunk(record, output_file)

        # -------- IMAGE CHUNKS --------
        # retrieval_text = LLaVA description
        # No alternative — images must be converted to text for retrieval
        print(f"   Describing {len(images)} image(s) with LLaVA...")
        for i, image in enumerate(images):
            print(f"   Image {i+1}/{len(images)}...", end=" ", flush=True)
            description = describe_image(image)
            print("done")

            record = {
                "chunk_id"      : str(uuid.uuid4()),
                "modality"      : "image",
                "source_pdf"    : filename,
                "page_number"   : None,
                "raw_text"      : None,
                "image_b64"     : image,
                "retrieval_text": description,
            }
            export_chunk(record, output_file)

        total_texts  += len(texts)
        total_tables += len(tables)
        total_images += len(images)

        print(f"   ✅ Done: {filename}")

    print(f"\n{'='*40}")
    print(f"INGESTION COMPLETE")
    print(f"  Text chunks  : {total_texts}")
    print(f"  Table chunks : {total_tables}")
    print(f"  Image chunks : {total_images}")
    print(f"  Output       : {output_file}")
    print(f"{'='*40}")


# ===============================
# 7. Entry Point
# ===============================

if __name__ == "__main__":
    pdf_directory = r"D:\Projects\Major_Project\documents"
    process_pdfs_in_directory(pdf_directory)
