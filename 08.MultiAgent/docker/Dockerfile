FROM python:3.11-slim

WORKDIR /app

# Backend
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ /app/backend/
COPY prompts/ /app/prompts/
COPY data/    /app/data/

WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
