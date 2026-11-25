# Cashly FAQ — Chatbot com RAG

Assistente virtual do **Cashly**, aplicativo de finanças pessoais. Ele responde dúvidas dos usuários usando **RAG** (Retrieval-Augmented Generation): em vez de responder de memória, o modelo consulta um FAQ do aplicativo e responde **apenas** com base nele.

Se a resposta não estiver no FAQ, o bot informa que não tem a informação e recomenda falar com o suporte humano.

## Como funciona

1. **Retrieve:** a pergunta do usuário é usada para buscar, no banco vetorial (Chroma), os blocos do FAQ mais parecidos.
2. **Augment:** os blocos encontrados são adicionados à pergunta como contexto.
3. **Generate:** o modelo de linguagem gera a resposta usando só esse contexto.

Perguntas de acompanhamento (ex.: "e se eu quiser mudar ela?") são reescritas como perguntas completas antes da busca, usando o histórico da conversa.

## Tecnologias

- Python 3
- [ChromaDB](https://www.trychroma.com/) — banco vetorial
- [OpenAI SDK](https://github.com/openai/openai-python) — cliente compatível com OpenAI e Groq
- [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn — API para uso no aplicativo
- python-dotenv — leitura do `.env`

## Estrutura do projeto

```
Cashly_FAQ/
├── .env.example        # modelo do arquivo de configuração
├── .gitignore
├── README.md
├── requirements.txt
├── config.py           # caminhos, nome da coleção e TOP_K
├── rag.py              # lógica do RAG (busca, reescrita e resposta)
├── index_faq.py        # indexa o FAQ no banco vetorial
├── chatbot.py          # chatbot no terminal
├── api.py              # API FastAPI (usada pelo app)
└── data/
    └── cashly_faq.txt  # base de conhecimento (FAQ)
```

## Como executar (terminal)

### 1. Instalar as dependências

```
pip install -r requirements.txt
```

### 2. Configurar a chave de API

Copie o arquivo `.env.example` para `.env` e preencha com a sua chave.

**Opção A — Groq:**

```
OPENAI_API_KEY=sua_chave_da_groq
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_CHAT_MODEL=openai/gpt-oss-20b
```

**Opção B — OpenAI:**

```
OPENAI_API_KEY=sua_chave_da_openai
OPENAI_CHAT_MODEL=gpt-4o-mini
```

> Na opção B, não use a linha `OPENAI_BASE_URL`.
> Se aparecer erro 404 de modelo, o modelo não está disponível na sua conta: troque o valor de `OPENAI_CHAT_MODEL`.

### 3. Indexar o FAQ

Cria o banco vetorial a partir de `data/cashly_faq.txt`:

```
python index_faq.py
```

### 4. Iniciar o chatbot

```
python chatbot.py
```

Digite sua pergunta e use `exit` (ou `sair`) para encerrar.

### Exemplo

```
You: o que é o Cashly?
Assistant: O Cashly é um aplicativo de finanças pessoais que permite controlar receitas e despesas...

You: qual a cotação do dólar?
Assistant: Desculpe, mas essa informação não está disponível no FAQ. Recomendo entrar em contato com o suporte humano...
```

## API (opcional)

A mesma lógica está disponível como API, usada pelo botão "?" do aplicativo Cashly.

```
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

A documentação interativa fica em `http://localhost:8000/docs`.

| Método | Rota      | Descrição                          |
|--------|-----------|------------------------------------|
| GET    | `/health` | Verifica se a API está no ar       |
| POST   | `/chat`   | Recebe uma pergunta e devolve a resposta |

**Exemplo de requisição para `/chat`:**

```json
{
  "message": "e se eu quiser mudar ela?",
  "history": [
    {"role": "user", "content": "como cadastro uma despesa?"},
    {"role": "assistant", "content": "Informe nome, valor, data e método de pagamento."}
  ]
}
```

**Resposta:**

```json
{
  "answer": "..."
}
```

A API não guarda estado: o aplicativo envia o histórico recente da conversa em cada chamada.

## Como editar o FAQ

O conteúdo do bot está em `data/cashly_faq.txt`. Cada bloco tem uma pergunta, a resposta logo abaixo e **uma linha em branco** separando do próximo:

```
Posso exportar meus gastos?
Resposta aqui, em um único parágrafo.

Outra pergunta?
Outra resposta.
```

Regras:

- Não deixe linha em branco dentro de um bloco.
- Um assunto por bloco.
- Escreva apenas o que o aplicativo realmente faz.

Depois de editar, rode novamente `python index_faq.py`.

## Casos de teste sugeridos

| Tipo | Pergunta | Resultado esperado |
|------|----------|--------------------|
| Dentro do FAQ | "O que é o Cashly?" | Descreve o aplicativo |
| Reformulada | "Dá pra separar meus gastos por tipo?" | Fala de categorias |
| Regra específica | "Quando recebo aviso de limite?" | 80% (alerta) e 100% (excedido) |
| Acompanhamento | "Como cadastro uma despesa?" → "E se eu quiser mudar ela?" | Fala de editar despesa |
| Fora do FAQ | "Qual a cotação do dólar?" | Recusa e indica o suporte humano |
| Fora do FAQ | "Quanto custa o Cashly?" | Recusa e indica o suporte humano |

## Limitações

- O bot só conhece o que está no `cashly_faq.txt`. Se o aplicativo mudar, o FAQ precisa ser atualizado.
- Modelos pequenos podem, ocasionalmente, adicionar detalhes que não estão no FAQ. O prompt do sistema instrui o modelo a não inventar, mas isso não é uma garantia absoluta.
- A busca usa o modelo de embeddings padrão do Chroma, que é focado em inglês. Se a busca em português errar os blocos, pode-se trocar por embeddings da OpenAI.

## Autores

- [Andressa Andrade](https://github.com/dressaDevv)
- [Alexander Emanuel](https://github.com/AlexSm121)
- [Joás Fortunato](https://github.com/JoasFortunato)
- [João Victor da Silva](https://github.com/JoaoVSilva-Devv)
- [José Elias](https://github.com/eliastk0)
- [Mariana Melo](https://github.com/marianameelo)
