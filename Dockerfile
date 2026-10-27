FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e ".[web]"

EXPOSE 8000

CMD ["uvicorn", "ai_document_assistant.web:app", "--host", "0.0.0.0", "--port", "8000"]
