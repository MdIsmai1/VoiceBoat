FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    netcat-openbsd \
    poppler-utils \
    tesseract-ocr \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python manage.py collectstatic --no-input

COPY wait-for-db.sh .
RUN chmod +x wait-for-db.sh

CMD ["sh", "-c", "./wait-for-db.sh && python manage.py makemigrations rag && python manage.py migrate && gunicorn --bind 0.0.0.0:${PORT:-10000} voice_pdf_rag.wsgi:application"]
