from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from bs4 import BeautifulSoup


# ===============================
# Utility
# ===============================

def html_table_to_text(html):
    """
    Convert HTML table to plain text before summarizing.
    gemma:2b is a text model — clean input gives better summaries.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        cells = [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


# ===============================
# Text + Table Summarizer
# ===============================

def summarize_texts_tables(texts, tables):
    """
    Summarize texts and tables using gemma:2b via Ollama.

    texts  : list of CompositeElement objects  (from unstructured)
    tables : list of Table objects             (from unstructured)

    Returns:
        text_summaries  : list of strings
        table_summaries : list of strings
    """

    prompt_text = """
Summarize the following table or text concisely in 1-2 sentences.
Do not add any commentary or phrases like "Here is the summary".

Input: {element}
"""

    prompt = ChatPromptTemplate.from_template(prompt_text)
    model  = ChatOllama(temperature=0.5, model="gemma:2b")
    chain  = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # FIX 1: extract .text from CompositeElement objects before passing to chain
    text_strings = [t.text if hasattr(t, "text") else str(t) for t in texts]
    raw_text_summaries = chain.batch(text_strings, {"max_concurrency": 3})
    text_summaries = [
        s if isinstance(s, str) else s.content for s in raw_text_summaries
    ]

    # FIX 2: convert HTML → plain text before summarizing
    # gemma:2b handles "col1 | col2" much better than raw <td> tags
    table_strings = [html_table_to_text(t.metadata.text_as_html) for t in tables]
    raw_table_summaries = chain.batch(table_strings, {"max_concurrency": 3})
    table_summaries = [
        s if isinstance(s, str) else s.content for s in raw_table_summaries
    ]

    return text_summaries, table_summaries


# ===============================
# Image Summarizer
# ===============================

def summarize_images(images_b64):
    """
    Summarize images using LLaVA via Ollama.

    images_b64 : list of base64-encoded image strings

    Returns:
        summaries : list of strings
    """

    llm    = ChatOllama(model="llava", temperature=0.0)
    parser = StrOutputParser()
    summaries = []

    for b64 in images_b64:

        data_url = f"data:image/jpeg;base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image accurately, in technical detail.",
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

        resp = llm.invoke(messages)

        # FIX 3: use parser.invoke() not parser.parse()
        # parse() expects a plain string; invoke() handles ChatMessage objects
        summaries.append(parser.invoke(resp))

    return summaries