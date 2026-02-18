import json
import uuid
import os
from bs4 import BeautifulSoup # this library is used to parse HTML tables into text
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, CompositeElement # we are importing these to check element types during segregation
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
        extract_image_block_to_payload=True,# this ensures we get the base64 string for each image block, which we need for LLaVA descriptions
        chunking_strategy="by_title", # this is the most semantic strategy — it creates chunks based on document structure (titles, sections).
        max_characters=10000,#
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
        if isinstance(el, Table): # here we are checking if the element is a Table type — this is how we identify tables for special handling later.
            tables.append(el)
        elif isinstance(el, CompositeElement):
            texts.append(el)
    return tables, texts
# this function iterates through all the elements returned by partition_pdf and separates them into two lists: one for tables and one for text. We check the type of each element — if it's a Table, it goes into the tables list; if it's a CompositeElement (which includes paragraphs, sections, etc.), it goes into the texts list. This segregation allows us to handle tables differently from regular text when we create our final chunks for retrieval.

def get_images(chunks):
    images_b64 = []
    for chunk in chunks:
        if isinstance(chunk, CompositeElement):
            chunk_els = chunk.metadata.orig_elements or [] # we look into the original elements that make up this chunk to find any image blocks. This is necessary because images are not top-level elements — they are nested inside CompositeElements. By accessing chunk.metadata.orig_elements, we can check if any of those original elements are images and extract their base64 strings for LLaVA descriptions later.
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64


# ===============================
# 3. Text Normalization
# ===============================

# this function takes raw text and normalizes it by collapsing multiple spaces into a single space. This is important for retrieval because we want to avoid issues with inconsistent spacing, which can affect both BM25 keyword matching and dense vector representations. By ensuring that all whitespace is standardized, we improve the chances that the retrieval systems will match queries to the relevant chunks effectively.
# for example- if the original text isi "This   is   a   test." (with multiple spaces), clean_text will convert it to "This is a test." (with single spaces). This makes the text cleaner and more consistent for retrieval.

def clean_text(text):
    return " ".join(text.split())


def html_table_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.find_all("tr"): # here findall looks for all table row elements in the HTML. Each 'tr' represents a row in the table.

        cells = [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])] # find.all looks for both 'td' (table data) and 'th' (table header) cells within each row. get_text(strip=True) extracts the text content of each cell and removes any leading/trailing whitespace. This way, we get a clean list of cell values for each row.

        if cells:
            rows.append(" | ".join(cells)) # this joins the cell values with " | " as a separator, creating a clear distinction between columns in the plain text format. Each row of the table becomes a single string with columns separated by " | ", and then all rows are joined together with newline characters to preserve the structure of the table in a readable format.
    return "\n".join(rows)

# for example, if we have a table with two rows and three columns, the HTML might look like this:
# <table>
#   <tr><th>Name</th><th>Age</th><th>City</th></tr>
#   <tr><td>Alice</td><td>30</td><td>New York</td></tr>
# </table>
# The html_table_to_text function would convert this to:
# Name | Age | City
# Alice | 30 | New York


# ===============================
# 4. Image Description via LLaVA
# ===============================

def describe_image(image_b64):
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


# function opens the output file in append mode and writes each chunk as a JSON line. This allows us to build up the chunks.jsonl file incrementally as we process each PDF, without needing to hold all chunks in memory at once. Each record is a dictionary containing the chunk_id, modality, source PDF, page number, raw text, base64 image (if applicable), and retrieval text. By writing each chunk immediately to the file, we ensure that we don't lose data if the process is interrupted and we can handle large numbers of chunks without memory issues.
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
        if not filename.lower().endswith(".pdf"): # we only want to process PDF files, so we check the file extension. If the file does not end with ".pdf" (case-insensitive), we skip it and move on to the next file in the directory. This ensures that we don't attempt to process non-PDF files, which would likely cause errors since our partitioning and chunking logic is designed specifically for PDFs.
            continue

        file_path = os.path.join(directory_path, filename) # join the directory path and filename to get the full path to the PDF file we want to process.
        elements  = create_chunks_from_pdf(file_path)

        print(f"\n📘 Processing: {filename}")
        print(f"   Elements found: {len(elements)}")

        tables, texts = table_text_segregation(elements)
        images= get_images(elements)

        print(f"   Texts: {len(texts)}, Tables: {len(tables)}, Images: {len(images)}")

        # -------- TEXT CHUNKS --------
        # retrieval_text = clean raw text
        # No summarisation — BM25 needs real keywords, dense needs real content

        for text in texts:
            raw_text = text.text if hasattr(text, "text") else str(text) #hasattr checks if the text element has a 'text' attribute.
            raw_text = clean_text(raw_text) # run the clean function defined earlier to normalize the text by collapsing multiple spaces into a single space. This ensures that the text is cleaner and more consistent for retrieval.

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
            print(f"   Image {i+1}/{len(images)}...", end=" ", flush=True) # this prints the progress of image description in the terminal. By using end=" " and flush=True, we ensure that the output is updated on the same line, giving a real-time progress update as each image is being processed by LLaVA. This is helpful because image description can take some time, and it lets the user know that the process is ongoing and how many images have been completed out of the total.
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
