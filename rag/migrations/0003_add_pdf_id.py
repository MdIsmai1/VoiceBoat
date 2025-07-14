from django.db import migrations, models
import uuid

class Migration(migrations.Migration):
    dependencies = [
        ('rag', '0002_alter_pdf_file_name'),
    ]
    operations = [
        migrations.AddField(
            model_name='pdf',
            name='pdf_id',
            field=models.UUIDField(default=uuid.uuid4, editable=False),
        ),
        migrations.AlterModelTable(
            name='conversationhistory',
            table=None,
        ),
        migrations.AlterModelTable(
            name='pdf',
            table=None,
        ),
        migrations.AlterModelTable(
            name='session',
            table=None,
        ),
    ]
