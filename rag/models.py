from django.db import models
import uuid

class Session(models.Model):
    session_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'session'

    def __str__(self):
        return str(self.session_id)

class Pdf(models.Model):
    pdf_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    file_name = models.CharField(max_length=255)
    pdf_hash = models.CharField(max_length=64)  # Assuming hash is a string (e.g., SHA256)
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name='pdfs')

    class Meta:
        db_table = 'pdf'

    def __str__(self):
        return self.file_name

class ConversationHistory(models.Model):
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name='conversations')
    question = models.TextField()
    answer = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'conversationhistory'

    def __str__(self):
        return f"{self.session.session_id} - {self.timestamp}"
