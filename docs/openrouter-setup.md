# OpenRouter setup

OpenRouter is the simplest provider for this project because one account can provide both chat-model and embedding requests through an OpenAI-compatible endpoint.

## Configure the project

From the project directory:

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

```text
AI_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=openrouter/free
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
```

Create the API key in your OpenRouter dashboard. Never commit `.env` or paste the key into source code or chat.

## Models

`openrouter/free` is convenient for learning, but free routing can have rate limits and changing model quality. When you want a stable portfolio demo, select a specific text model from the [OpenRouter model catalog](https://openrouter.ai/models) and put its provider/slug identifier in `OPENROUTER_MODEL`.

Embeddings are separate from answer generation. The example uses `openai/text-embedding-3-small`, which OpenRouter documents as an embeddings model. Embedding requests are billed according to the selected model, so indexing a large book may use credits.

## Test the configuration

The project tests do not call OpenRouter. To make one direct request yourself:

```powershell
python -m ai_document_assistant "Explain attention in one paragraph."
```

To index a PDF and ask grounded questions:

```powershell
python -m ai_document_assistant `
  --database data\assistant.db `
  --max-pages 3 `
  --pdf "your-document.pdf" `
  "What is the main idea?"
```

The `--max-pages 3` option keeps the first experiment small. The first command uses the chat model. The second uses the embedding model while indexing and the chat model while answering.
