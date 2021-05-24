from django.db import models

# Create your models here.
class Tryon(models.Model):
    pose = models.ImageField(upload_to="image")
    cloth = models.ImageField(upload_to="cloth")

    def __str__(self):
        return self.id