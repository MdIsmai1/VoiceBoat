from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [
        ('rag', '0001_initial'),
    ]
    operations = [
        migrations.AlterField(
            model_name='Pdf',
            name='file_name',
            field=models.CharField(max_length=255, null=True, blank=True),
        ),
    ]
