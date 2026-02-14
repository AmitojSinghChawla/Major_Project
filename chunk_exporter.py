# Chunks exporter py where we export the chunks in a jsonl format with the following fields:
# - chunk_id: unique identifier for the chunk
# - modality: "text", "image", or "table"
# - source_pdf: name of the source PDF file
# - page_number: page number in the PDF (if available)
# - raw_text: the raw text content for text/table chunks (if available)
# - image_b64: base64 string for image chunks (if available)
# - gold_questions: list of gold questions associated with this chunk (initially empty)

import json
import uuid
import os
from unstructured.partition.pdf import partition_pdf

def create_chunks_from_pdf(file_path):

    elements= partition_pdf(
        filename=file_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,# it is used to convert the image into base64 fromat to used in web apps and apis.
        chunking_strategy="by_title", # this allows chunking to become a part of the partiioning The by_title chunking strategy preserves section boundaries and optionally page boundaries as well. “Preserving” here means that a single chunk will never contain text that occurred in two different sections. When a new section starts, the existing chunk is closed and a new one started, even if the next element would fit in the prior chunk. In addition to the behaviors of the basic strategy above, the by_title strategy has the following behaviors: 1. Detect Section Headings, 2. Respect page Boundaries, 3. combine small sections
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

    return elements

def process_pdfs_in_directory(directory_path):

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
           file_path=os.path.join(directory_path, filename) # returns the full path of the file by joining the directory path and the filename
           elements=create_chunks_from_pdf(file_path)
           yield filename, elements

           # this is a generator function that yields a tuple of the filename and the elements for each PDF file in the directory. It allows us to process each PDF one at a time without loading all PDFs into memory at once.


def table_text_segregation(all_elements):
    """
    Segregate elements into tables and texts.
    """
    tables = []
    texts = []

    for el in all_elements:
        el_type = str(type(el))
        if "Table" in el_type:
            tables.append(el)
        elif "CompositeElement" in el_type:
            texts.append(el)

    return tables, texts


def get_images(chunks):
    """
    Extract images from CompositeElements.
    """
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)): #
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

# set([str(type(el)) for el in elements])
# {"<class 'unstructured.documents.elements.CompositeElement'>"}
# this is the unique set of element types that we get from the partition pdf function. We can see that we have only one type of element which is CompositeElement. This means that all the elements in the PDF are being grouped together into a single CompositeElement. We can use this information to further analyze the structure of the PDF and extract the relevant information from the CompositeElement.

# <unstructured.documents.elements.Title at 0x1adf8f2a610>,
#  <unstructured.documents.elements.NarrativeText at 0x1adf8f28690>,
#  <unstructured.documents.elements.Footer at 0x1adf8f2a450>,
#  <unstructured.documents.elements.Text at 0x1ad84ebb0d0>,
#  <unstructured.documents.elements.Image at 0x1adf8c233d0>,

# this is the unique set of element types that we get from the orig_elements of the CompositeElement. We can see that we have different types of elements such as Title, NarrativeText, Footer, Text, and Image. This means that the CompositeElement is grouping together different types of elements from the PDF. We can use this information to further analyze the structure of the PDF and extract the relevant information from each type of element.


def export_text_chunks(texts, source_pdf, output_file="chunks.jsonl"):

    with open(output_file, "a", encoding="utf-8") as f:
        for text in texts:
            raw_text=text.text if hasattr(text, "text") else str(text)

            record= {
                "chunk_id": str(uuid.uuid4()),
                "modality" : "text",
                "source_pdf": source_pdf,
                "page_number": None,
                "raw_text": raw_text,
                "image_b64": None,
                "gold_questions": []
            }
            f.write(json.dumps(record) + "\n")


def export_image_chunks(images_b64, source_pdf, output_file="chunks.jsonl"):
    with open(output_file, "a", encoding="utf-8") as f:
        for image in images_b64:
            record= {
                "chunk_id": str(uuid.uuid4()),
                "modality" : "image",
                "source_pdf": source_pdf,
                "page_number": None,
                "raw_text": None,
                "image_b64": image,
                "gold_questions": []
            }

            f.write(json.dumps(record) + "\n")

def export_table_chunks(tables, source_pdf, output_file="chunks.jsonl"):
    with open(output_file, "a", encoding="utf-8") as f:
        for table in tables:
            raw_text=None
            if hasattr(table, "metadata") and hasattr(table.metadata, "text_as_html"):
                raw_text=table.metadata.text_as_html
            else:
                raw_text=str(table)
                # this is a fallback option in case we don't have a good textual representation of the table. It will convert the table object into a string format which may not be ideal but at least gives us some representation of the table content.

            record= {
                "chunk_id": str(uuid.uuid4()),
                "modality" : "table",
                "source_pdf": source_pdf,
                "page_number": None,
                "raw_text": raw_text,
                "image_b64": None,
                "gold_questions": []
            }

            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    pdf_directory = r"D:\Projects\Major_Project\documents"

    for pdf_name, elements in process_pdfs_in_directory(pdf_directory):
        print(f"\n📘 Processing: {pdf_name}")
        print(f"Ingesting {pdf_name} with {len(elements)} elements")

        # Segregate tables, texts, and images
        tables, texts = table_text_segregation(elements)
        images = get_images(elements)

        print(
            f"Texts: {len(texts)}, "
            f"Tables: {len(tables)}, "
            f"Images: {len(images)}"
        )

        # Export chunks
        export_text_chunks(texts, pdf_name)
        export_image_chunks(images, pdf_name)
        export_table_chunks(tables, pdf_name)

        print(f"Exported chunks for {pdf_name} to chunks.jsonl")
