FROM python:3.11

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    tesseract-ocr \
    libtesseract-dev \
    && apt-get clean

# Upgrade pip to avoid dependency resolution issues
RUN pip install --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Collect static files
RUN python manage.py collectstatic --no-input

# Run migrations
RUN python run_migrations.py

# Start Gunicorn with PORT environment variable
CMD gunicorn voice_pdf_rag.wsgi:application --bind 0.0.0.0:$PORT
