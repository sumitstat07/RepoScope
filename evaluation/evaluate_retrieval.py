import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.retriever import RAGRetriever
from src.config import EVAL_DIR

K_VALUES = [1, 3, 5]


def main():
    df = pd.read_csv(EVAL_DIR / "test_questions.csv")
    retriever = RAGRetriever()

    rows = []
    for _, row in df.iterrows():
        question = str(row["question"]).strip()
        expected = str(row["expected_source"]).strip()

        hits = retriever.retrieve(question, top_k=max(K_VALUES))
        retrieved = [h["source_file"] for h in hits]

        record = {
            "question": question,
            "expected_source": expected,
            "top_1": retrieved[0] if retrieved else "",
            "retrieved": " | ".join(retrieved),
        }
        for k in K_VALUES:
            record[f"hit@{k}"] = int(expected in retrieved[:k])
        rows.append(record)

    results = pd.DataFrame(rows)
    out_path = EVAL_DIR / "results.csv"
    results.to_csv(out_path, index=False)

    n = len(results)
    print("\n=== RETRIEVAL EVALUATION ===")
    print(f"test questions: {n}")
    for k in K_VALUES:
        hits = int(results[f"hit@{k}"].sum())
        print(f"Recall@{k}: {hits}/{n} = {hits / n:.1%}")

    misses = results[results["hit@3"] == 0]
    if len(misses):
        print("\nMissed at k=3:")
        for _, m in misses.iterrows():
            print(f"  - {m['question'][:70]}  (expected {m['expected_source']})")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()


    