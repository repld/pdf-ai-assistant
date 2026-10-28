# OpenRouter setup

OpenRouter provides both chat-model and embedding requests through an OpenAI-compatible endpoint.

## Configure the project

From the project directory:

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

```text
AI_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=deepseek/deepseek-v3.2
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
```

Create the API key in your OpenRouter dashboard. Never commit `.env` or paste the key into source code or chat.

## Models

`deepseek/deepseek-v3.2` is the default low-cost model configured for this application. OpenRouter may route it through multiple providers, so check the model page for current pricing and availability before deployment.

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

You can select a specific range instead:

```powershell
python -m ai_document_assistant `
  --database data\assistant.db `
  --page-start 30 `
  --page-end 32 `
  --pdf "book.pdf" `
  "What do these pages discuss?"
```
