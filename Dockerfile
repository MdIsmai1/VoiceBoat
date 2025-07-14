FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    netcat-openbsd \
    poppler-utils \
    tesseract-ocr \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to avoid compatibility issues
RUN pip install --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Collect static files
RUN python manage.py collectstatic --no-input --verbosity 2

# Copy and make wait-for-db.sh executable
COPY wait-for-db.sh .
RUN chmod +x wait-for-db.sh

# Run migrations with verbose output and database check
CMD ["sh", "-c", "./wait-for-db.sh && python manage.py makemigrations rag --verbosity 3 && python manage.py migrate --verbosity 3 && gunicorn --bind 0.0.0.0:${PORT:-10000} voice_pdf_rag.wsgi:application --log-level debug"]
