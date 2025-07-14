from django.contrib import admin
from .models import Pdf, ConversationHistory, Session

@admin.register(Pdf)
class PdfAdmin(admin.ModelAdmin):
    list_display = ['file_name', 'pdf_hash', 'session']
    list_filter = ['session']

@admin.register(ConversationHistory)
class ConversationHistoryAdmin(admin.ModelAdmin):
    list_display = ['session', 'question', 'answer', 'timestamp']
    list_filter = ['session', 'timestamp']

@admin.register(Session)
class SessionAdmin(admin.ModelAdmin):
    list_display = ['session_id', 'created_at', 'last_activity']
    list_filter = ['created_at', 'last_activity']
