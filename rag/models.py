from django.db import models
import uuid

class Pdf(models.Model):
    pdf_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    # Other fields (add your existing fields here, e.g., file_name, etc.)
    file_name = models.CharField(max_length=255)  # Example field, adjust as needed

    class Meta:
        db_table = 'pdf'  # Optional: ensures table name is 'pdf'

# Other models (e.g., ConversationHistory, Session) can remain unchanged
class ConversationHistory(models.Model):
    # Add your fields here
    pass

    class Meta:
        db_table = 'conversationhistory'

class Session(models.Model):
    # Add your fields here
    pass

    class Meta:
        db_table = 'session'
