# from django.db import models
# import uuid

# class Session(models.Model):
#     session_id = models.CharField(max_length=36, primary_key=True, default=uuid.uuid4)
#     created_at = models.DateTimeField(auto_now_add=True)
#     last_activity = models.DateTimeField(auto_now=True)

#     class Meta:
#         db_table = 'sessions'

# class Pdf(models.Model):
#     pdf_id = models.CharField(max_length=36, primary_key=True, default=uuid.uuid4)
#     session = models.ForeignKey(Session, on_delete=models.CASCADE)
#     pdf_path = models.CharField(max_length=512)
#     pdf_hash = models.CharField(max_length=64)
#     uploaded_at = models.DateTimeField(auto_now_add=True)

#     class Meta:
#         db_table = 'pdfs'

# class ConversationHistory(models.Model):
#     session = models.ForeignKey(Session, on_delete=models.CASCADE)

#     timestamp = models.DateTimeField(auto_now_add=True)
#     question = models.TextField()
#     answer = models.TextField()

#     class Meta:
#         db_table = 'conversation_history'






from django.db import models
import uuid

class Session(models.Model):
    session_id = models.UUIDField(default=uuid.uuid4, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = 'rag'

class Pdf(models.Model):
    pdf_id = models.UUIDField(default=uuid.uuid4, unique=True)
    file_name = models.CharField(max_length=255)
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
