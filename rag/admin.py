from django.contrib import admin
from .models import Session, Pdf, ConversationHistory

@admin.register(Session)
class SessionAdmin(admin.ModelAdmin):
    list_display = ('session_id', 'created_at', 'last_activity')
    search_fields = ('session_id',)
    list_filter = ('created_at', 'last_activity')

@admin.register(Pdf)
class PdfAdmin(admin.ModelAdmin):
    list_display = ('file_name', 'pdf_hash', 'session')
    search_fields = ('file_name', 'pdf_hash')
    list_filter = ('session',)

@admin.register(ConversationHistory)
class ConversationHistoryAdmin(admin.ModelAdmin):
    list_display = ('session', 'question', 'answer', 'timestamp')
    search_fields = ('question', 'answer')
    list_filter = ('session', 'timestamp')
