FROM python:3.11
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y ffmpeg tesseract-ocr
RUN pip install --no-cache-dir -r requirements.txt
RUN python manage.py collectstatic --no-input
ENV PORT=8000
CMD gunicorn voice_pdf_rag.wsgi:application --bind 0.0.0.0:$PORT
