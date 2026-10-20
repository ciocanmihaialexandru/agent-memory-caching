"""Demo prompt caching Anthropic (cache_control: ephemeral).

Pune system prompt + context fix (documentul CEC) în cache și măsoară
tokens economisiți + reducerea de latență între apelul 1 (write) și apelul 2 (read).
"""

import time

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from extraction.loaders import load_document

load_dotenv()

MODEL = "claude-sonnet-4-5"       # model Anthropic cu suport de caching (schimbă dacă ai alt acces)
DOC_PATH = "data/cec_comisioane.pdf"


def build_messages(context: str, question: str) -> list:
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Ești un asistent bancar care răspunde pe baza documentului CEC."},
                {
                    "type": "text",
                    "text": context,                           # contextul FIX (documentul)
                    "cache_control": {"type": "ephemeral"},    # <-- breakpoint de cache
                },
            ],
        },
        {"role": "user", "content": question},
    ]


def call(model, messages):
    start = time.perf_counter()
    resp = model.invoke(messages)
    elapsed_ms = (time.perf_counter() - start) * 1000
    details = resp.usage_metadata["input_token_details"]
    return elapsed_ms, details


def main():
    context = load_document(DOC_PATH)   # textul complet al documentului = contextul fix
    model = ChatAnthropic(model=MODEL, temperature=0)

    print(f"Context: {len(context)} caractere\n")

    # Apel 1 — SCRIE cache-ul
    t1, d1 = call(model, build_messages(context, "Ce pachete de cont are CEC Bank?"))
    print(f"Apel 1 (write): {t1:7.0f} ms  |  cache_creation={d1.get('cache_creation', 0)}  "
          f"ephemeral_5m={d1.get('ephemeral_5m_input_tokens', 0)}  cache_read={d1.get('cache_read', 0)}")

    # Apel 2 — CITEȘTE cache-ul (altă întrebare, ACELAȘI context cached)
    t2, d2 = call(model, build_messages(context, "Cât costă retragerea de la ATM la pachetul PREMIUM?"))
    print(f"Apel 2 (read):  {t2:7.0f} ms  |  cache_creation={d2.get('cache_creation', 0)}  "
          f"ephemeral_5m={d2.get('ephemeral_5m_input_tokens', 0)}  cache_read={d2.get('cache_read', 0)}")

    print("\n--- Rezumat ---")
    print(f"Tokens citiți din cache la apelul 2: {d2.get('cache_read', 0)}")
    if t1 > 0:
        print(f"Reducere latență apel 2 vs 1: {t1 - t2:.0f} ms ({(t1 - t2) / t1 * 100:.0f}%)")


if __name__ == "__main__":
    main()