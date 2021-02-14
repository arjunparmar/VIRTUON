from django.db import models

# Create your models here.
class Tryon(models.Model):
    pose = models.ImageField(upload_to="model/input/test/test/image")
    cloth = models.ImageField(upload_to="model/input/test/test/cloth")

    def __str__(self):
        return self.id