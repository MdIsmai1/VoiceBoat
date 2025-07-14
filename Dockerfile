FROM python:3.11

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    tesseract-ocr \
    libtesseract-dev \
    netcat-openbsd \
    && apt-get clean

# Upgrade pip
RUN pip install --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Collect static files
RUN python manage.py collectstatic --no-input

# Wait for database and run migrations
COPY wait-for-db.sh .
RUN chmod +x wait-for-db.sh
RUN ./wait-for-db.sh && python manage.py migrate --noinput || { echo "Migration failed"; exit 1; }

# Start Gunicorn
CMD gunicorn voice_pdf_rag.wsgi:application --bind 0.0.0.0:$PORT --workers 2 --timeout 120
