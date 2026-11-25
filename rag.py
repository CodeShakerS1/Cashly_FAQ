"""RAG logic shared by the terminal chatbot and the API."""
import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from config import COLLECTION_NAME, TOP_K, VECTOR_DB_PATH

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
except Exception:
    raise RuntimeError("Vector database not found. Run 'python index_faq.py' first.")

SYSTEM_PROMPT = (
    "Você é o assistente virtual do aplicativo Cashly, de controle financeiro pessoal. "
    "Responda às dúvidas usando APENAS as informações do FAQ fornecido. "
    "Não invente nomes de telas, menus, botões, passos nem exemplos que não estejam "
    "escritos no FAQ. "
    "Se a pergunta for genérica ou vaga (por exemplo 'como funciona?', 'o que é isso?', "
    "'ajuda', 'me explica') ou for apenas um cumprimento, não diga que falta informação: "
    "explique em poucas frases o que é o Cashly e sugira 3 ou 4 perguntas que o usuário "
    "pode fazer, usando só o que está no FAQ. "
    "Se a pergunta for específica e a resposta não estiver no FAQ, diga educadamente que "
    "não possui essa informação e recomende falar com o suporte humano."
)

MAX_HISTORY = 6  # last messages sent to the model


def rewrite_question(question, history):
    """Turns a follow-up like 'and if I want to change it?' into a standalone question."""
    previous = history[-4:]
    if not previous:
        return question

    conversation = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in previous)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Reescreva a última pergunta do usuário como uma pergunta completa "
                    "e independente, usando o contexto da conversa. Se ela já for "
                    "independente, repita-a sem mudanças. Responda apenas com a pergunta."
                ),
            },
            {
                "role": "user",
                "content": f"Conversa:\n{conversation}\n\nÚltima pergunta: {question}",
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def answer_question(question, history):
    """history is a list of {"role": "user" | "assistant", "content": "..."}."""
    history = history[-MAX_HISTORY:]

    # R (Retrieve)
    search_query = rewrite_question(question, history)
    result = collection.query(query_texts=[search_query], n_results=TOP_K)
    context = "\n---\n".join(result["documents"][0])

    print("SEARCH QUERY:", search_query)
    print("CONTEXT:", context)

    # A (Augment)
    augmented_question = f"Contexto do FAQ:\n{context}\n\nPergunta: {search_query}"

    # G (Generate)
    response = client.chat.completions.create(
        model=model,
        messages=(
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + history
            + [{"role": "user", "content": augmented_question}]
        ),
        temperature=0,
    )
    return response.choices[0].message.content