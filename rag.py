"""RAG logic shared by the terminal chatbot and the API."""
import os
import re
import unicodedata

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from config import COLLECTION_NAME, FAQ_PATH, TOP_K, VECTOR_DB_PATH

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
DEBUG = os.getenv("DEBUG_RAG") == "1"

chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
except Exception:
    raise RuntimeError("Vector database not found. Run 'python index_faq.py' first.")

try:
    FAQ_BLOCKS = [
        block.strip()
        for block in FAQ_PATH.read_text(encoding="utf-8").split("\n\n")
        if block.strip()
    ]
except FileNotFoundError:
    raise RuntimeError(f"FAQ file not found: {FAQ_PATH}")

SYSTEM_PROMPT = (
    "Você é o assistente virtual do aplicativo Cashly, de controle financeiro pessoal. "
    "Responda às dúvidas usando APENAS as informações do FAQ fornecido. "
    "Não invente nomes de telas, menus, botões, passos nem exemplos que não estejam "
    "escritos no FAQ. "
    "Responda em no máximo 3 frases, em texto puro, sem markdown, sem asteriscos e "
    "sem listas numeradas. Não repita a mesma informação em frases diferentes. "
    "Se a pergunta for apenas um cumprimento, cumprimente e sugira 3 perguntas copiadas "
    "literalmente do FAQ. "
    "Se a pergunta for específica e a resposta não estiver no FAQ, diga educadamente que "
    "não possui essa informação e recomende falar com o suporte humano."
)

MAX_HISTORY = 6  # last messages sent to the model

# Words present in almost every FAQ block: they carry no search signal.
STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "para", "pra", "pelo", "pela", "com", "sem",
    "e", "ou", "que", "qual", "quais", "como", "onde", "quando", "porque", "porque",
    "se", "eu", "me", "meu", "minha", "meus", "minhas", "ele", "ela", "eles", "elas",
    "isso", "isto", "esse", "essa", "este", "esta", "aquele", "aquela", "la", "ai",
    "e", "sao", "ser", "esta", "estao", "ter", "tem", "tenho", "posso", "pode",
    "fazer", "faz", "consigo", "quero", "queria", "mais", "menos", "muito", "muita",
    "ja", "nao", "sim", "ao", "aos", "cashly", "app", "aplicativo", "funciona",
    "funcionam", "existe", "existem", "tem", "ha",
}


def _keywords(text):
    """Lowercase, strip accents/punctuation and drop low-signal words."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return {w for w in text.split() if len(w) > 2 and w not in STOPWORDS}


def _lexical_score(query_words, block):
    """How much of the query appears in the block (question line weighs more)."""
    if not query_words:
        return 0.0
    question = block.partition("\n")[0]
    question_score = len(query_words & _keywords(question)) / len(query_words)
    block_score = len(query_words & _keywords(block)) / len(query_words)
    return 0.6 * question_score + 0.4 * block_score


def _retrieve(search_query, original_question):
    queries = list(dict.fromkeys(q for q in (search_query, original_question) if q))
    query_words = _keywords(" ".join(queries))

    distances = {}
    for query in queries:
        result = collection.query(query_texts=[query], n_results=max(TOP_K * 3, 10))
        for document, distance in zip(result["documents"][0], result["distances"][0]):
            distances[document] = min(distance, distances.get(document, float("inf")))

    # Rescue blocks found only by keywords (strong matches, not already retrieved)
    lexical = sorted(
        ((_lexical_score(query_words, block), block) for block in FAQ_BLOCKS),
        key=lambda item: item[0],
        reverse=True,
    )
    for score, block in lexical[:3]:
        if score >= 0.5 and block not in distances:
            distances[block] = 1.2  # neutral distance for keyword-only hits

    ranked = sorted(
        distances.items(),
        key=lambda item: item[1] - 1.5 * _lexical_score(query_words, item[0]),
    )
    return [block for block, _ in ranked[:TOP_K]]


def rewrite_question(question, history):
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
                    "e independente em português do Brasil, usando o contexto da conversa. "
                    "Se ela já for independente, repita-a sem mudanças. "
                    "Responda apenas com a pergunta, sem aspas e sem explicações."
                ),
            },
            {
                "role": "user",
                "content": f"Conversa:\n{conversation}\n\nÚltima pergunta: {question}",
            },
        ],
        temperature=0,
    )
    rewritten = (response.choices[0].message.content or "").strip().strip('"').strip()
    rewritten = re.sub(r"^(pergunta|question)\s*:\s*", "", rewritten, flags=re.IGNORECASE)
    return rewritten or question


def answer_question(question, history):
    history = history[-MAX_HISTORY:]

    # R (Retrieve)
    search_query = rewrite_question(question, history)
    blocks = _retrieve(search_query, question)
    context = "\n---\n".join(blocks)

    if DEBUG:
        print(f"\n[debug] pergunta reescrita: {search_query}")
        for i, block in enumerate(blocks, 1):
            print(f"[debug] bloco {i}: {block.partition(chr(10))[0]}")

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
    answer = response.choices[0].message.content or ""
    answer = answer.replace("**", "")
    return re.sub(r"\n{3,}", "\n\n", answer).strip()