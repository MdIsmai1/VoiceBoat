from django.db import migrations, models
import uuid

class Migration(migrations.Migration):
    initial = True
    dependencies = []
    operations = [
        migrations.CreateModel(
            name='Session',
            fields=[
                ('session_id', models.UUIDField(default=uuid.uuid4, primary_key=True, serialize=False)),
                ('django_session_key', models.CharField(max_length=40, null=True, unique=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('last_activity', models.DateTimeField(auto_now=True)),
            ],
            options={'app_label': 'rag'},
        ),
        migrations.CreateModel(
            name='Pdf',
            fields=[
                ('pdf_id', models.UUIDField(default=uuid.uuid4, primary_key=True, serialize=False)),
                ('file_name', models.CharField(blank=True, max_length=255, null=True)),
                ('pdf_hash', models.CharField(max_length=64)),
                ('session', models.ForeignKey(on_delete=models.CASCADE, to='rag.session')),
            ],
            options={'app_label': 'rag'},
        ),
        migrations.CreateModel(
            name='ConversationHistory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question', models.TextField()),
                ('answer', models.TextField()),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('session', models.ForeignKey(on_delete=models.CASCADE, to='rag.session')),
            ],
            options={'app_label': 'rag'},
        ),
    ]
