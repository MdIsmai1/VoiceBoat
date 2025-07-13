Voice PDF RAG Web Application (Django)
A Django-based web application for uploading PDFs, asking questions via voice or text, and receiving audio responses using a Retrieval-Augmented Generation (RAG) pipeline. Supports multiple users with isolated conversation histories.
Prerequisites

Python 3.8–3.11
Poppler and Tesseract OCR installed and added to PATH
Ollama with llama3.2 model
Internet access for gTTS and speech recognition

Setup

Create the project structure:voice_pdf_rag_web/
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── main.js
├── templates/
│   └── index.html
├── voice_pdf_rag/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── rag/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── migrations/
│   │   └── __init__.py
│   ├── models.py
│   ├── tests.py
│   └── views.py
├── manage.py
├── requirements.txt
└── README.md


Create and activate a virtual environment:python -m venv venv
.\venv\Scripts\activate


Install dependencies:pip install -r requirements.txt


Install Poppler and Tesseract:
Poppler: Download from https://github.com/oschwartz10612/poppler-windows
Tesseract: Download from https://github.com/UB-Mannheim/tesseract/wiki
Add to PATH (e.g., C:\Program Files\poppler-24.07.0\bin, C:\Program Files\Tesseract-OCR)


Install and run Ollama:ollama pull llama3.2
ollama run llama3.2


(Optional) Add a favicon.ico to static/ to suppress favicon errors.

Running the Application

Apply database migrations:python manage.py makemigrations
python manage.py migrate


Collect static files:python manage.py collectstatic


Start the Django server:python manage.py runserver 7862


Open http://localhost:7862 in your browser (do not use VS Code Live Server).
Upload a PDF, ask questions via microphone or text, and receive audio responses.

Troubleshooting

404 Not Found: Ensure index.html is in templates/ and the Django server is running (python manage.py runserver 7862).
Unexpected end of JSON input: Check browser console (F12) and Django logs for errors. Test /ask/ with:curl -X POST -F "session_id=test" -F "text_input=What is the main topic?" http://localhost:7862/ask/


Kernel crashes: Use small PDFs (<1MB), ensure sufficient RAM, and verify Poppler/Tesseract:tesseract --version
pdftoppm --version


Permission issues: Grant write access to %temp%:icacls %temp% /grant Users:F /T


Dependency issues: Reinstall dependencies:pip install -r requirements.txt --force-reinstall



Features

Upload PDFs (max 10MB, non-encrypted)
Ask questions via microphone or text
Receive audio responses (gTTS)
View per-user conversation history (toggleable)
Reset session
Multi-user support via Django sessions
