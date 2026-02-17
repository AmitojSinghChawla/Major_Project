"""
07_chatbot.py
-------------
Simple terminal chatbot over your ingested PDFs.

Uses hybrid + reranker (best retrieval method) to find relevant chunks,
then passes them to Ollama (gemma:2b) to generate an answer.

This is a demo — not part of core evaluation.

Usage:
    python 07_chatbot.py
    python 07_chatbot.py --method hybrid    # use a different retrieval method
    python 07_chatbot.py --top_k 3          # retrieve fewer chunks
"""

import argparse
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from retrieve import load_indexes, retrieve


DEFAULT_METHOD = "hybrid_reranker"
DEFAULT_TOP_K  = 5


# ===============================
# Answer generation
# ===============================

def generate_answer(question, retrieved_chunks):
    """
    Build context from retrieved chunks and generate answer.
    Shows which modality each context piece came from.
    """

    context_parts = []

    for chunk in retrieved_chunks:
        label = f"[{chunk['modality'].upper()} | {chunk['source_pdf']}]"

        if chunk["modality"] == "image":
            context_parts.append(f"{label}\n{chunk['retrieval_text']}")
        elif chunk["raw_text"]:
            context_parts.append(f"{label}\n{chunk['raw_text']}")
        else:
            context_parts.append(f"{label}\n{chunk['retrieval_text']}")

    context = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions based on document excerpts.
Answer using only the context provided.
If the context does not contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:""")

    model = ChatOllama(model="gemma:2b", temperature=0.1)
    chain = prompt | model | StrOutputParser()

    return chain.invoke({"context": context, "question": question})


# ===============================
# Show retrieved sources
# ===============================

def show_sources(retrieved_chunks):
    print("\n  Sources used:")
    for chunk in retrieved_chunks:
        print(f"    [{chunk['rank']}] {chunk['modality']:6s} | "
              f"{chunk['source_pdf']} | "
              f"score={chunk['score']:.4f} | "
              f"chunk_id={chunk['chunk_id'][:8]}...")


# ===============================
# Main chat loop
# ===============================

def main(method, top_k):

    print("Loading indexes...")
    indexes = load_indexes()

    print(f"\nChatbot ready — method: {method.upper()}, top_k: {top_k}")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    while True:
        print("─" * 50)
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        # Retrieve relevant chunks
        retrieved = retrieve(question, method=method, indexes=indexes, top_k=top_k)

        if not retrieved:
            print("Bot: No relevant chunks found.")
            continue

        # Show sources
        show_sources(retrieved)

        # Generate answer
        print("\nBot:", end=" ", flush=True)
        answer = generate_answer(question, retrieved)
        print(answer)
        print()


# ===============================
# Entry point
# ===============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        default=DEFAULT_METHOD,
        choices=["bm25", "dense", "hybrid", "hybrid_reranker"],
        help="Retrieval method to use"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve"
    )
    args = parser.parse_args()

    main(args.method, args.top_k)
