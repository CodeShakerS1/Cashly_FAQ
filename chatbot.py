"""Cashly RAG chatbot (terminal).

    python chatbot.py
"""
try:
    from rag import answer_question
except RuntimeError as error:
    raise SystemExit(str(error))

history = []

print("=" * 60)
print("Cashly chatbot started! Type your question or 'exit' to quit.")
print("=" * 60)

while True:
    question = input("\nYou: ").strip()

    if question.lower() in ("exit", "sair"):
        print("Bye!")
        break

    if not question:
        continue

    answer = answer_question(question, history)

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})

    print(f"\nAssistant: {answer}")