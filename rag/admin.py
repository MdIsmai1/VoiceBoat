from django.contrib import admin
from .models import Session, Pdf, ConversationHistory

@admin.register(Session)
class SessionAdmin(admin.ModelAdmin):
    list_display = ('session_id', 'last_activity')
    search_fields = ('session_id',)

@admin.register(Pdf)
class PdfAdmin(admin.ModelAdmin):
    list_display = ('pdf_path', 'pdf_hash', 'session')
    search_fields = ('pdf_path', 'pdf_hash')

@admin.register(ConversationHistory)
class ConversationHistoryAdmin(admin.ModelAdmin):
    list_display = ('session', 'question', 'answer', 'timestamp')
    search_fields = ('question', 'answer')
    list_filter = ('timestamp', 'session')