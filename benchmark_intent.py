"""Benchmark: Intent Classifier (sklearn) vs LLM — latență, cost, accuracy."""

import time

from dotenv import load_dotenv

from classifier.predict import predict_intent

load_dotenv()

LABELS = ["search", "extract", "summarize"]

# set de test etichetat (întrebări NOI, cu răspunsul corect)
TEST_SET = [
    ("Cât costă emiterea cardului la INFINITE?", "search"),
    ("Ce taxă are plata online?", "search"),
    ("Care e comisionul lunar la SMART?", "search"),
    ("Listează comisioanele pachetului PENSIONAR", "extract"),
    ("Dă-mi toate taxele pentru transferuri", "extract"),
    ("Enumeră pachetele pentru studenți", "extract"),
    ("Rezumă oferta pentru pensionari", "summarize"),
    ("Pe scurt, ce avantaje are INFINITE?", "summarize"),
    ("Fă-mi un sumar al documentului", "summarize"),
]


def bench_classifier() -> dict:
    correct = 0
    start = time.perf_counter()
    for q, gold in TEST_SET:
        if str(predict_intent(q)) == gold:
            correct += 1
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {"accuracy": correct / len(TEST_SET),
            "avg_ms": elapsed_ms / len(TEST_SET),
            "cost_usd": 0.0}


def bench_llm() -> dict:
    from langchain_anthropic import ChatAnthropic
    model = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
    system = ("Ești un clasificator de intenții. Răspunde DOAR cu un cuvânt dintre: "
              "search, extract, summarize. Fără explicații, fără punctuație.")

    correct = 0
    total_ms = 0.0
    in_tokens = out_tokens = 0
    for q, gold in TEST_SET:
        start = time.perf_counter()
        resp = model.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": q},
        ])
        total_ms += (time.perf_counter() - start) * 1000

        raw = resp.content.strip().lower()
        pred = next((lbl for lbl in LABELS if lbl in raw), raw)  # robust la "intenția e search"
        if pred == gold:
            correct += 1

        u = resp.usage_metadata
        in_tokens += u["input_tokens"]
        out_tokens += u["output_tokens"]

    # preț aproximativ Sonnet: $3 / 1M input, $15 / 1M output
    cost = in_tokens / 1_000_000 * 3 + out_tokens / 1_000_000 * 15
    return {"accuracy": correct / len(TEST_SET),
            "avg_ms": total_ms / len(TEST_SET),
            "cost_usd": cost}


def main() -> None:
    print(f"Test set: {len(TEST_SET)} întrebări\n")

    clf = bench_classifier()
    print(f"CLASSIFIER  acc={clf['accuracy']:.0%}  latență={clf['avg_ms']:.2f} ms/req   cost=$0 (local)")

    llm = bench_llm()
    print(f"LLM         acc={llm['accuracy']:.0%}  latență={llm['avg_ms']:.0f} ms/req   cost=${llm['cost_usd']:.4f} ({len(TEST_SET)} req)")

    print("\n--- Comparație ---")
    print(f"Latență: clasificatorul e ~{llm['avg_ms'] / clf['avg_ms']:.0f}× mai rapid")
    print(f"Cost:    clasificatorul $0 vs LLM ${llm['cost_usd']:.4f} pentru {len(TEST_SET)} cereri")
    print(f"Accuracy: clasificator {clf['accuracy']:.0%} vs LLM {llm['accuracy']:.0%}")


if __name__ == "__main__":
    main()