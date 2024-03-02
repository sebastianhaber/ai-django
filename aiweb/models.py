from django.db import models
from django.core.files.storage import default_storage
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from django.core.files.base import ContentFile
import numpy as np



# Create your models here.
class Article(models.Model):
    title = models.CharField(max_length=255, blank=True)
    content = models.TextField(blank=True)
    photo = models.ImageField(upload_to="mediaphoto", blank=True, null=True)

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

        if self.photo:
            try:
                file_path = self.photo.path
                if default_storage.exists(file_path):
                    pil_image = tf_image.load_img(file_path, target_size=(299, 299))
                    img_array = tf_image.img_to_array(pil_image)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)

                    model = InceptionV3(weights='imagenet')
                    predictions = model.predict(img_array)
                    decodedPredictions = decode_predictions(predictions, top=1)[0]
                    bestGuess = decodedPredictions[0][1]
                    self.title = bestGuess
                    self.content = ', '.join([f"{pred[1]}: {pred[2] * 100:.2f}%" for pred in decodedPredictions])

                    super().save(*args, **kwargs)
            except Exception as e:
                pass
