"""
06_evaluate.py
--------------
Unified evaluation — retrieval metrics + RAGAS generation metrics.

For every question in gold_questions.json:
  1. Run all 4 retrieval methods
  2. Compute Recall@k, Precision@k, MRR, nDCG@k from ranked chunk_ids
  3. Pass top chunks to Ollama (gemma) to generate an answer
  4. Score the answer with RAGAS (Faithfulness, Answer Relevancy, etc.)

Results saved to evaluation_results.json and printed as a summary table.

Usage:
    python 06_evaluate.py
"""

import json
import math
import os
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from retrieve import load_indexes, retrieve


QUESTIONS_FILE = "gold_questions.json"
RESULTS_FILE   = "evaluation_results.json"

METHODS = ["bm25", "dense", "hybrid", "hybrid_reranker"]
TOP_K   = 5

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")   # set this in your environment


# ===============================
# Load gold questions
# ===============================

def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ===============================
# Retrieval metrics
# ===============================

def recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved_top_k = [r["chunk_id"] for r in retrieved_ids[:k]]
    hits = len(set(retrieved_top_k) & set(relevant_ids))
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids, relevant_ids, k):
    retrieved_top_k = [r["chunk_id"] for r in retrieved_ids[:k]]
    hits = len(set(retrieved_top_k) & set(relevant_ids))
    return hits / k


def mrr(retrieved_ids, relevant_ids):
    """
    Mean Reciprocal Rank — 1/rank of first correct result.
    Perfect retrieval = 1.0 (correct chunk at rank 1)
    """
    for result in retrieved_ids:
        if result["chunk_id"] in relevant_ids:
            return 1.0 / result["rank"]
    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    """
    nDCG@k — rewards putting relevant chunks higher in the ranking.
    """
    retrieved_top_k = [r["chunk_id"] for r in retrieved_ids[:k]]

    # DCG — actual ranking
    dcg = 0.0
    for i, cid in enumerate(retrieved_top_k):
        if cid in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)   # i+2 because log2(1) = 0

    # IDCG — perfect ranking (all relevant at top)
    idcg = 0.0
    for i in range(min(len(relevant_ids), k)):
        idcg += 1.0 / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(results, relevant_ids, k=TOP_K):
    return {
        f"recall@{k}"   : round(recall_at_k(results, relevant_ids, k), 4),
        f"precision@{k}": round(precision_at_k(results, relevant_ids, k), 4),
        "mrr"           : round(mrr(results, relevant_ids), 4),
        f"ndcg@{k}"     : round(ndcg_at_k(results, relevant_ids, k), 4),
    }


# ===============================
# Generate answer with Ollama
# ===============================

def generate_answer(question, retrieved_chunks):
    """
    Build context from top retrieved chunks and generate an answer.
    Handles text, table, and image chunks (images use retrieval_text as proxy).
    """

    context_parts = []
    for chunk in retrieved_chunks:
        if chunk["modality"] == "image":
            # For images we use the LLaVA description as context
            context_parts.append(f"[Image Description]: {chunk['retrieval_text']}")
        elif chunk["raw_text"]:
            context_parts.append(chunk["raw_text"])
        else:
            context_parts.append(chunk["retrieval_text"])

    context = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
Be concise. If the context does not contain the answer, say "Not found in context."

Context:
{context}

Question: {question}

Answer:""")

    model  = ChatOllama(model="gemma:2b", temperature=0.0)
    chain  = prompt | model | StrOutputParser()

    return chain.invoke({"context": context, "question": question})


# ===============================
# RAGAS evaluation
# ===============================

def run_ragas(questions, answers, contexts, ground_truths):
    """
    questions     : list of question strings
    answers       : list of generated answer strings
    contexts      : list of lists — each inner list is the retrieved texts for that question
    ground_truths : list of ground truth answer strings
    """

    data = {
        "question"    : questions,
        "answer"      : answers,
        "contexts"    : contexts,
        "ground_truth": ground_truths,
    }

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    return result


# ===============================
# Main evaluation loop
# ===============================

def main():

    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set. RAGAS will fail.")
        print("Set it with:  export OPENAI_API_KEY=your_key_here\n")

    questions = load_questions()
    print(f"Loaded {len(questions)} gold questions\n")

    indexes = load_indexes()

    all_results = {}   # method → list of per-question results

    for method in METHODS:
        print(f"\n{'='*50}")
        print(f"Evaluating method: {method.upper()}")
        print(f"{'='*50}")

        method_results = []

        # Collect for RAGAS
        ragas_questions     = []
        ragas_answers       = []
        ragas_contexts      = []
        ragas_ground_truths = []

        for q in questions:
            question       = q["question"]
            relevant_ids   = q["relevant_chunk_ids"]
            ground_truth   = q["ground_truth"]

            # --- Retrieve ---
            retrieved = retrieve(question, method=method, indexes=indexes, top_k=TOP_K)

            # --- Retrieval metrics ---
            retrieval_metrics = compute_retrieval_metrics(retrieved, relevant_ids, k=TOP_K)

            print(f"\n  Q: {question}")
            print(f"  Retrieval: {retrieval_metrics}")

            # --- Generate answer ---
            answer = generate_answer(question, retrieved)
            print(f"  Answer: {answer[:100]}...")

            # --- Collect for RAGAS ---
            ragas_questions.append(question)
            ragas_answers.append(answer)
            ragas_contexts.append([c["retrieval_text"] for c in retrieved])
            ragas_ground_truths.append(ground_truth)

            method_results.append({
                "question_id"     : q["question_id"],
                "question"        : question,
                "modality"        : q["modality"],
                "retrieved_ids"   : [r["chunk_id"] for r in retrieved],
                "retrieval_metrics": retrieval_metrics,
                "generated_answer": answer,
                "ground_truth"    : ground_truth,
            })

        # --- Run RAGAS for this method ---
        print(f"\n  Running RAGAS for {method}...")
        ragas_scores = run_ragas(
            ragas_questions,
            ragas_answers,
            ragas_contexts,
            ragas_ground_truths,
        )

        # Attach RAGAS scores to results
        # RAGAS v0.2+ returns an EvaluationResult — .to_pandas() still works
        ragas_df = ragas_scores.to_pandas()
        for i, row in ragas_df.iterrows():
            def safe(val):
                # RAGAS returns NaN when a metric fails — guard against round(NaN) crash
                import math
                return round(float(val), 4) if val is not None and not math.isnan(float(val)) else None

            method_results[i]["ragas_metrics"] = {
                "faithfulness"     : safe(row.get("faithfulness")),
                "answer_relevancy" : safe(row.get("answer_relevancy")),
                "context_precision": safe(row.get("context_precision")),
                "context_recall"   : safe(row.get("context_recall")),
            }

        all_results[method] = method_results

    # Save full results
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {RESULTS_FILE}")

    # Print summary table
    print_summary(all_results)


# ===============================
# Print summary table
# ===============================

def print_summary(all_results):

    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")

    for method, results in all_results.items():
        n = len(results)
        if n == 0:
            continue

        # Average retrieval metrics
        avg_recall    = sum(r["retrieval_metrics"][f"recall@{TOP_K}"]    for r in results) / n
        avg_precision = sum(r["retrieval_metrics"][f"precision@{TOP_K}"] for r in results) / n
        avg_mrr       = sum(r["retrieval_metrics"]["mrr"]                 for r in results) / n
        avg_ndcg      = sum(r["retrieval_metrics"][f"ndcg@{TOP_K}"]      for r in results) / n

        # Average RAGAS metrics (may not exist if RAGAS failed)
        ragas_results = [r for r in results if "ragas_metrics" in r]
        if ragas_results:
            def avg_metric(key):
                vals = [r["ragas_metrics"][key] for r in ragas_results if r["ragas_metrics"].get(key) is not None]
                return sum(vals) / len(vals) if vals else 0.0

            avg_faith = avg_metric("faithfulness")
            avg_rel   = avg_metric("answer_relevancy")
            avg_cp    = avg_metric("context_precision")
            avg_cr    = avg_metric("context_recall")
        else:
            avg_faith = avg_rel = avg_cp = avg_cr = 0.0

        print(f"\nMethod: {method.upper()}")
        print(f"  Retrieval")
        print(f"    Recall@{TOP_K}     : {avg_recall:.4f}")
        print(f"    Precision@{TOP_K}  : {avg_precision:.4f}")
        print(f"    MRR           : {avg_mrr:.4f}")
        print(f"    nDCG@{TOP_K}      : {avg_ndcg:.4f}")
        print(f"  Generation (RAGAS)")
        print(f"    Faithfulness      : {avg_faith:.4f}")
        print(f"    Answer Relevancy  : {avg_rel:.4f}")
        print(f"    Context Precision : {avg_cp:.4f}")
        print(f"    Context Recall    : {avg_cr:.4f}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
