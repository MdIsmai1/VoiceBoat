import os
import django
import time
from django.db.utils import OperationalError
from django.core.management import call_command

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'voice_pdf_rag.settings')
retries = 5
for i in range(retries):
    try:
        django.setup()
        call_command('migrate')
        print("Migrations applied successfully")
        break
    except OperationalError as e:
        print(f"Database connection failed: {e}. Retrying in 5 seconds...")
        time.sleep(5)
else:
    print("Failed to apply migrations after retries")
    exit(1)
