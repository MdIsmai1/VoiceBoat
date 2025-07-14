from django.db import models
import uuid

class Session(models.Model):
    session_id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    django_session_key = models.CharField(max_length=40, unique=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = 'rag'

class Pdf(models.Model):
    pdf_id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    file_name = models.CharField(max_length=255, null=True, blank=True)  # Reintroduced to match migration
    pdf_hash = models.CharField(max_length=64)
    session = models.ForeignKey(Session, on_delete=models.CASCADE)

    class Meta:
        app_label = 'rag'

class ConversationHistory(models.Model):
    session = models.ForeignKey(Session, on_delete=models.CASCADE)
    question = models.TextField()
    answer = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'rag'
