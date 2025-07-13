import os
import re
import uuid
import json
import logging
import tempfile
import hashlib
import asyncio
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import warnings
import numpy as np
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from django.http import JsonResponse, HttpResponse, FileResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.middleware.csrf import get_token
from asgiref.sync import sync_to_async, async_to_sync
from gtts import gTTS
import speech_recognition as sr
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from datetime import datetime, timedelta
import soundfile as sf
from pydub import AudioSegment
from .models import Session, Pdf, ConversationHistory

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize global models and cache
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2")
VOICE_IDS = {'male': 'en-US'}
vector_db_cache: Dict[str, Chroma] = {}
vector_db_cache_lock = Lock()
executor = ThreadPoolExecutor(max_workers=10)  # Increased for multi-user support

def check_dependencies():
    required_modules = ['gtts', 'speech_recognition', 'PyPDF2', 'pdf2image', 'pytesseract', 'langchain_core', 'langchain_ollama', 'langchain_community', 'numpy', 'soundfile', 'chromadb', 'pydub']
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        raise ImportError(f"Missing dependencies: {', '.join(missing)}")
    try:
        pytesseract.get_tesseract_version()
        logger.info("Tesseract OCR is installed.")
    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract OCR is not installed or not in PATH.")
        raise RuntimeError("Tesseract OCR is not installed or not in PATH.")

async def cleanup_sessions(max_age_hours: int = 24):
    current_time = datetime.now()
    expired_time = current_time - timedelta(hours=max_age_hours)
    expired_sessions = await sync_to_async(lambda: list(Session.objects.filter(last_activity__lt=expired_time)))()
    for session in expired_sessions:
        with vector_db_cache_lock:
            for key in list(vector_db_cache.keys()):
                if key.startswith(str(session.session_id)):
                    try:
                        vector_db_cache[key].delete_collection()
                        del vector_db_cache[key]
                    except Exception as e:
                        logger.warning(f"Error deleting vector DB for session {session.session_id}: {str(e)}")
        await sync_to_async(Pdf.objects.filter(session=session).delete)()
        await sync_to_async(ConversationHistory.objects.filter(session=session).delete)()
        await sync_to_async(session.delete)()
        logger.info(f"Cleaned up expired session {session.session_id}")

async def process_pdf(pdf_file: str, session_id: str) -> Tuple[Optional[Chroma], List[str]]:
    try:
        pdf_path = Path(pdf_file)
        logger.debug(f"Processing PDF: {pdf_file} for session {session_id}")
        if not isinstance(pdf_file, str) or not pdf_file.strip():
            return None, ["Failed: Invalid PDF file path."]
        if not pdf_path.exists():
            return None, ["Failed: PDF file does not exist."]
        if pdf_path.suffix.lower() != '.pdf':
            return None, ["Failed: Invalid file format. Please upload a PDF."]
        if pdf_path.stat().st_size > 10 * 1024 * 1024:
            return None, ["Failed: PDF file too large (max 10MB)."]

        with open(pdf_path, "rb") as f:
            pdf_hash = hashlib.sha256(f.read()).hexdigest()

        pdf_exists = await sync_to_async(lambda: Pdf.objects.filter(pdf_hash=pdf_hash, session_id=session_id).exists())()
        if pdf_exists:
            with vector_db_cache_lock:
                cache_key = f"{session_id}:{pdf_hash}"
                if cache_key in vector_db_cache:
                    logger.info(f"Using cached vector database for {pdf_path.name} in session {session_id}")
                    return vector_db_cache[cache_key], []

        logger.info(f"Processing PDF: {pdf_path.name} for session {session_id}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_pdf_path = Path(tmp_dir) / pdf_path.name
            with open(pdf_path, "rb") as src, open(tmp_pdf_path, "wb") as dst:
                dst.write(src.read())
            
            reader = PdfReader(tmp_pdf_path)
            if reader.is_encrypted:
                return None, ["Failed: PDF is encrypted."]
            if len(reader.pages) == 0:
                return None, ["Failed: PDF has no pages."]

            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                text += extracted or ""
            logger.info(f"Extracted {len(text)} characters with PyPDF2")

            ocr_text = ""
            try:
                images = convert_from_path(tmp_pdf_path, first_page=1, last_page=5)
                for i, image in enumerate(images):
                    ocr_result = pytesseract.image_to_string(image, lang='eng')
                    ocr_text += ocr_result or ""
                    logger.debug(f"OCR extracted {len(ocr_result)} characters from page {i + 1}")
            except Exception as e:
                logger.warning(f"OCR processing failed: {str(e)}. Falling back to text extraction.")

            combined_text = text + "\n" + ocr_text
            if not combined_text.strip():
                return None, ["Failed: No text extracted from PDF."]
            data = [Document(page_content=combined_text, metadata={"source": str(pdf_path)})]

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            chunks = text_splitter.split_documents(data)

            persist_dir = Path(tempfile.gettempdir()) / f"chroma_{session_id}_{pdf_hash[:8]}"
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=f"pdf_collection_{pdf_hash[:8]}_{session_id}",
                persist_directory=str(persist_dir)
            )
            with vector_db_cache_lock:
                vector_db_cache[f"{session_id}:{pdf_hash}"] = vector_db
            
            session = await sync_to_async(lambda: Session.objects.get(session_id=session_id))()
            await sync_to_async(Pdf.objects.create)(
                session=session,
                pdf_path=str(pdf_path),
                pdf_hash=pdf_hash
            )
            
            return vector_db, []
    except Exception as e:
        logger.error(f"Error processing PDF for session {session_id}: {str(e)}")
        return None, [f"Failed: Error processing PDF: {str(e)}"]

async def chat_with_pdf(question: str, vector_db: Optional[Chroma], session_id: str) -> str:
    try:
        if vector_db is None:
            logger.error(f"Vector database is None for session {session_id}")
            return "Error: Vector database not initialized. Please upload a valid PDF."
        
        template = """You are an assistant for question-answering tasks. Use the following context to answer the question accurately and concisely in 1-2 sentences. If you don't know the answer, say so and do not make up information.

        Context: {context}

        Question: {question}

        Answer: """
        qa_prompt = ChatPromptTemplate.from_template(template)

        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(search_kwargs={"k": 3}),
            llm,
            prompt=PromptTemplate.from_template("Generate multiple variations of this question: {question}")
        )

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        answer = await rag_chain.ainvoke(question)
        session = await sync_to_async(lambda: Session.objects.get(session_id=session_id))()
        await sync_to_async(ConversationHistory.objects.create)(
            session=session,
            question=question,
            answer=answer
        )
        logger.info(f"Answered question: '{question}' for session {session_id}")
        return answer
    except Exception as e:
        logger.error(f"Error in RAG pipeline for session {session_id}: {str(e)}")
        return f"Error processing question: {str(e)}."

async def save_tts_audio(text: str, audio_response_path: str, voice_gender: str = "male") -> None:
    try:
        os.makedirs(os.path.dirname(audio_response_path), exist_ok=True)
        max_chunk_length = 500
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_chunk_length:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
        logger.info(f"Split text into {len(chunks)} chunks for TTS")

        chunk_files = []
        language = VOICE_IDS.get(voice_gender.lower(), 'en-US')
        for i, chunk in enumerate(chunks):
            chunk_path = f"{audio_response_path}_{i}.mp3"
            tts = gTTS(text=chunk, lang=language, slow=False)
            tts.save(chunk_path)
            logger.info(f"Generated gTTS audio chunk {i+1}/{len(chunks)} at {chunk_path}")
            chunk_files.append(chunk_path)

        if chunk_files:
            converted_files = []
            for i, chunk_file in enumerate(chunk_files):
                wav_path = f"{audio_response_path}_{i}.wav"
                data, samplerate = sf.read(chunk_file)
                sf.write(wav_path, data, samplerate)
                converted_files.append(wav_path)
                os.remove(chunk_file)
                logger.info(f"Converted MP3 to WAV: {wav_path}")
            chunk_files = converted_files

        if len(chunk_files) > 1:
            combined_data = []
            samplerate = None
            for chunk_file in chunk_files:
                data, sr = sf.read(chunk_file)
                if samplerate is None:
                    samplerate = sr
                combined_data.append(data)
            combined_data = np.concatenate(combined_data)
            sf.write(audio_response_path, combined_data, samplerate)
            for chunk_file in chunk_files:
                os.remove(chunk_file)
            logger.info(f"Concatenated audio chunks into {audio_response_path}")
        elif chunk_files:
            os.rename(chunk_files[0], audio_response_path)
            logger.info(f"Renamed single audio chunk to {audio_response_path}")
    except Exception as e:
        logger.error(f"Failed to generate audio: {str(e)}")
        raise

def index(request):
    logger.debug("Serving index.html")
    if not request.session.session_key:
        request.session.create()
    request.session.modified = True
    return render(request, 'index.html', {'csrf_token': get_token(request), 'session_id': request.session.session_key})

@csrf_exempt
@async_to_sync
async def ask(request):
    logger.debug("Received /ask request")
    try:
        if not request.session.session_key:
            request.session.create()
            request.session.modified = True
        session_id = request.session.session_key
        logger.debug(f"Session ID: {session_id}")

        # Update session last_activity
        session = await sync_to_async(lambda: Session.objects.get_or_create(session_id=session_id, defaults={'last_activity': datetime.now()})[0])()
        await sync_to_async(lambda: Session.objects.filter(session_id=session_id).update(last_activity=datetime.now()))()

        pdf_file = request.FILES.get('pdf_file')
        pdf_path = None
        if pdf_file:
            filename = pdf_file.name
            pdf_path = settings.MEDIA_ROOT / f"{session_id}_{filename}"
            with open(pdf_path, 'wb+') as destination:
                for chunk in pdf_file.chunks():
                    destination.write(chunk)
            logger.debug(f"Saved PDF to {pdf_path} for session {session_id}")

        status = "Processing PDF..."
        vector_db, errors = await process_pdf(str(pdf_path) if pdf_path else "", session_id)
        if vector_db is None:
            logger.error(f"PDF processing failed for session {session_id}: {errors[0] if errors else 'Unknown error'}")
            return JsonResponse({'status': errors[0] if errors else "Failed: Unable to process PDF."}, status=400)

        question = ""
        if 'audio_data' in request.FILES:
            audio_file = request.FILES['audio_data']
            audio_path = settings.MEDIA_ROOT / f"{session_id}_{audio_file.name}"
            with open(audio_path, 'wb+') as destination:
                for chunk in audio_file.chunks():
                    destination.write(chunk)
            logger.debug(f"Saved audio to {audio_path} for session {session_id}")
            status = "Transcribing audio..."
            try:
                temp_wav_path = settings.MEDIA_ROOT / f"converted_{session_id}_{uuid.uuid4().hex[:8]}.wav"
                audio = AudioSegment.from_file(audio_path)
                audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                audio.export(temp_wav_path, format="wav")
                logger.debug(f"Converted audio to WAV: {temp_wav_path}")

                recognizer = sr.Recognizer()
                with sr.AudioFile(str(temp_wav_path)) as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = recognizer.record(source)
                question = recognizer.recognize_google(audio_data)
                logger.info(f"Recognized: '{question}' for session {session_id}")
                os.remove(audio_path)
                os.remove(temp_wav_path)
            except sr.UnknownValueError:
                logger.error(f"Could not understand audio for session {session_id}")
                return JsonResponse({'status': "Failed: Could not understand audio."}, status=400)
            except Exception as e:
                logger.error(f"Transcription error for session {session_id}: {str(e)}")
                return JsonResponse({'status': f"Failed: Transcription error: {str(e)}."}, status=500)
        elif request.POST.get('text_input'):
            question = request.POST['text_input']
            logger.info(f"Using text input: '{question}' for session {session_id}")
        else:
            logger.error(f"No audio or text input provided for session {session_id}")
            return JsonResponse({'status': "Failed: No audio or text input provided."}, status=400)

        status = "Generating response..."
        answer = await chat_with_pdf(question, vector_db, session_id)
        if answer.startswith("Error"):
            logger.error(f"RAG pipeline error for session {session_id}: {answer}")
            return JsonResponse({'status': f"Failed: {answer}"}, status=500)

        status = "Generating audio response..."
        try:
            audio_response_path = Path(tempfile.gettempdir()) / f"response_{session_id}_{uuid.uuid4().hex[:8]}.wav"
            await save_tts_audio(answer, str(audio_response_path))
            os.chmod(audio_response_path, 0o600)
            logger.info(f"Audio response generated at {audio_response_path} for session {session_id}")
            return JsonResponse({'audio_path': f'/audio/{audio_response_path.name}', 'status': "Query processed successfully!"})
        except Exception as e:
            logger.error(f"Error generating audio response for session {session_id}: {str(e)}")
            return JsonResponse({'status': f"Failed: Error generating audio response: {str(e)}. Text response: {answer}"}, status=500)
    except Exception as e:
        logger.error(f"Unexpected error in /ask for session {session_id}: {str(e)}")
        return JsonResponse({'status': f"Failed: Unexpected error: {str(e)}"}, status=500)

@csrf_exempt
@async_to_sync
async def summary(request):
    logger.debug("Received /summary request")
    try:
        if not request.session.session_key:
            request.session.create()
            request.session.modified = True
        session_id = request.session.session_key
        logger.debug(f"Session ID: {session_id}")

        # Update session last_activity
        await sync_to_async(lambda: Session.objects.get_or_create(session_id=session_id, defaults={'last_activity': datetime.now()})[0])()
        await sync_to_async(lambda: Session.objects.filter(session_id=session_id).update(last_activity=datetime.now()))()

        data = json.loads(request.body.decode('utf-8'))
        is_visible = data.get('is_visible', False)
        
        if is_visible:
            history = await sync_to_async(lambda: list(ConversationHistory.objects.filter(session__session_id=session_id).order_by('timestamp')))()
            if not history:
                logger.info(f"No conversation history available for session {session_id}")
                return JsonResponse({'summary': "No conversation history available.", 'is_visible': True})
            summary = "<h3 class='text-lg font-semibold text-gray-800 mb-2'>Conversation History</h3>"
            for entry in history:
                summary += f"<p class='mb-2'><strong class='text-gray-700'>Timestamp:</strong> {entry.timestamp}<br><strong class='text-gray-700'>Question:</strong> {entry.question}<br><strong class='text-gray-700'>Answer:</strong> {entry.answer}</p><hr class='border-gray-200 my-2'>"
            logger.info(f"Returning conversation summary for session {session_id}")
            return JsonResponse({'summary': summary, 'is_visible': True})
        logger.info(f"Hiding conversation summary for session {session_id}")
        return JsonResponse({'summary': "", 'is_visible': False})
    except Exception as e:
        logger.error(f"Error in /summary for session {session_id}: {str(e)}")
        return JsonResponse({'summary': f"Failed: Error retrieving conversation history: {str(e)}", 'is_visible': False}, status=500)

@csrf_exempt
@async_to_sync
async def reset(request):
    logger.debug("Received /reset request")
    try:
        if not request.session.session_key:
            request.session.create()
            request.session.modified = True
        session_id = request.session.session_key
        logger.debug(f"Resetting session {session_id}")

        # Update session last_activity
        await sync_to_async(lambda: Session.objects.get_or_create(session_id=session_id, defaults={'last_activity': datetime.now()})[0])()
        await sync_to_async(lambda: Session.objects.filter(session_id=session_id).update(last_activity=datetime.now()))()

        # Clear all session-related data
        with vector_db_cache_lock:
            for key in list(vector_db_cache.keys()):
                if key.startswith(str(session_id)):
                    try:
                        vector_db_cache[key].delete_collection()
                        del vector_db_cache[key]
                        logger.info(f"Deleted vector DB for session {session_id}: {key}")
                    except Exception as e:
                        logger.warning(f"Error deleting vector DB for session {session_id}: {str(e)}")
        await sync_to_async(Pdf.objects.filter(session__session_id=session_id).delete)()
        await sync_to_async(ConversationHistory.objects.filter(session__session_id=session_id).delete)()
        await sync_to_async(Session.objects.filter(session_id=session_id).delete)()

        # Create a new session
        request.session.flush()  # Clear the current session
        request.session.create()
        request.session.modified = True
        new_session_id = request.session.session_key
        await sync_to_async(Session.objects.create)(session_id=new_session_id, last_activity=datetime.now())
        logger.info(f"Created new session {new_session_id} after reset")

        return JsonResponse({'status': "Bot reset successfully!", 'session_id': new_session_id})
    except Exception as e:
        logger.error(f"Error in /reset for session {session_id}: {str(e)}")
        return JsonResponse({'status': f"Failed: Error resetting conversation: {str(e)}"}, status=500)

def serve_audio(request, filename):
    logger.debug(f"Serving audio file: {filename}")
    try:
        return FileResponse(open(Path(tempfile.gettempdir()) / filename, 'rb'), content_type='audio/wav')
    except Exception as e:
        logger.error(f"Error serving audio file {filename}: {str(e)}")
        return JsonResponse({'status': f"Failed: Error serving audio: {str(e)}"}, status=500)
    
def upload_pdf(request):
    if request.method == 'POST' and request.FILES.get('pdf'):
        pdf_file = request.FILES['pdf']
        file_path = default_storage.save(f'uploads/{pdf_file.name}', pdf_file)
        # Process PDF
        return HttpResponse(f"Uploaded: {file_path}")
    return HttpResponse("Invalid request")
