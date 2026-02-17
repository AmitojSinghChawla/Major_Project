"""
05_annotate.py
--------------
Helps you build your gold_questions.json file for evaluation.

You run this script, it shows you chunks one by one so you can browse them,
then you type your questions and chunk_ids manually into gold_questions.json.

Alternatively — just edit gold_questions.json directly by hand.
That is perfectly fine for a bachelor's thesis.

gold_questions.json format:
[
  {
    "question_id"       : "q1",
    "question"          : "What was the revenue in Q3?",
    "relevant_chunk_ids": ["abc-123"],
    "ground_truth"      : "Revenue in Q3 was $5.2 million.",
    "modality"          : "table"   <- which modality does the answer live in?
  },
  ...
]

Usage:
    python 05_annotate.py --browse        # browse chunks to find chunk_ids
    python 05_annotate.py --add           # add a new question interactively
    python 05_annotate.py --show          # show all questions added so far
"""

import json
import argparse
import os


CHUNKS_FILE    = "chunks.jsonl"
QUESTIONS_FILE = "gold_questions.json"


# ===============================
# Load helpers
# ===============================

def load_chunks():
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_questions():
    if not os.path.exists(QUESTIONS_FILE):
        return []
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_questions(questions):
    with open(QUESTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)


# ===============================
# Browse chunks
# ===============================

def browse_chunks(filter_modality=None):
    """
    Print chunks to terminal so you can find the right chunk_id for your question.
    """
    chunks = load_chunks()

    if filter_modality:
        chunks = [c for c in chunks if c["modality"] == filter_modality]
        print(f"\nShowing {len(chunks)} chunks with modality='{filter_modality}'\n")
    else:
        print(f"\nShowing all {len(chunks)} chunks\n")

    for i, chunk in enumerate(chunks):
        print(f"{'─'*60}")
        print(f"Index        : {i}")
        print(f"chunk_id     : {chunk['chunk_id']}")
        print(f"modality     : {chunk['modality']}")
        print(f"source_pdf   : {chunk['source_pdf']}")
        print(f"page_number  : {chunk['page_number']}")
        print(f"retrieval_text (first 300 chars):")
        print(f"  {chunk['retrieval_text'][:300]}")
        print()

        # Pause every 5 chunks so terminal doesn't flood
        if (i + 1) % 5 == 0:
            cont = input("Press Enter to see more, or q to quit: ")
            if cont.strip().lower() == "q":
                break


# ===============================
# Add a question interactively
# ===============================

def add_question():
    questions = load_questions()

    print("\nAdd a new gold question")
    print("(You should have already browsed chunks to find the correct chunk_id)\n")

    question_id = f"q{len(questions) + 1}"

    question = input("Question: ").strip()
    if not question:
        print("Question cannot be empty.")
        return

    chunk_ids_input = input("Relevant chunk_id(s) — comma separated: ").strip()
    relevant_chunk_ids = [c.strip() for c in chunk_ids_input.split(",") if c.strip()]
    if not relevant_chunk_ids:
        print("Must provide at least one chunk_id.")
        return

    ground_truth = input("Ground truth answer (what the correct answer is): ").strip()
    if not ground_truth:
        print("Ground truth cannot be empty — RAGAS needs this.")
        return

    modality = input("Modality of the answer chunk (text / table / image): ").strip()
    if modality not in ("text", "table", "image"):
        print(f"Warning: '{modality}' is not a standard modality. Saving anyway.")

    record = {
        "question_id"       : question_id,
        "question"          : question,
        "relevant_chunk_ids": relevant_chunk_ids,
        "ground_truth"      : ground_truth,
        "modality"          : modality,
    }

    questions.append(record)
    save_questions(questions)

    print(f"\nSaved question '{question_id}' to {QUESTIONS_FILE}")
    print(json.dumps(record, indent=2))


# ===============================
# Show all questions
# ===============================

def show_questions():
    questions = load_questions()

    if not questions:
        print(f"\nNo questions yet in {QUESTIONS_FILE}")
        print("Run:  python 05_annotate.py --add")
        return

    print(f"\n{len(questions)} gold question(s) in {QUESTIONS_FILE}\n")

    for q in questions:
        print(f"{'─'*60}")
        print(f"ID             : {q['question_id']}")
        print(f"Question       : {q['question']}")
        print(f"Chunk IDs      : {q['relevant_chunk_ids']}")
        print(f"Ground Truth   : {q['ground_truth']}")
        print(f"Modality       : {q['modality']}")
    print()


# ===============================
# Entry point
# ===============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--browse", action="store_true",  help="Browse chunks to find chunk_ids")
    parser.add_argument("--add",    action="store_true",  help="Add a new gold question")
    parser.add_argument("--show",   action="store_true",  help="Show all gold questions")
    parser.add_argument("--modality", type=str, default=None, help="Filter browse by modality: text | table | image")
    args = parser.parse_args()

    if args.browse:
        browse_chunks(filter_modality=args.modality)
    elif args.add:
        add_question()
    elif args.show:
        show_questions()
    else:
        parser.print_help()
