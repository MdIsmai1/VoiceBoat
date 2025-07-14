from django.db import migrations, models
import uuid

class Migration(migrations.Migration):
    dependencies = [
        ('rag', '0003_add_pdf_id'),
    ]
    operations = [
        migrations.RemoveField(
            model_name='pdf',
            name='id',
        ),
        migrations.AlterField(
            model_name='pdf',
            name='pdf_id',
            field=models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False),
        ),
    ]
