from django.shortcuts import render
from django.http import JsonResponse

import tensorflow as tf
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import json
from django.views.decorators.csrf import csrf_exempt
# Load trained model
model = tf.keras.models.load_model('model.keras')


def home(request):
    return render(request, 'index.html')

@csrf_exempt
def predict_digit(request):

    if request.method == 'POST':

        data = json.loads(request.body)

        image_data = data['image']

        # Remove metadata
        image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)

        image = Image.open(BytesIO(image_bytes)).convert('L')

        # Resize to MNIST size
        image = image.resize((28,28))

        image = np.array(image)

        # Normalize
        image = image / 255.0

        # Reshape
        image = image.reshape(1,28,28,1)

        prediction = np.argmax(model.predict(image))

        return JsonResponse({
            'prediction': int(prediction)
        })