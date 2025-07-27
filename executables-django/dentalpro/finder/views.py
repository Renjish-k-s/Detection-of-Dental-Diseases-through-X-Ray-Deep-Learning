import os
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.core.files.storage import default_storage
from tensorflow.keras.preprocessing import image

# Load the trained NASNet Large model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dlmodel/nasnet_large_model_pakka.h5')
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
class_labels = {0: 'Cavity', 1: 'Filling', 2: 'Impacted Tooth', 3: 'Implant'}

def preprocess_image(image_path):
    """Preprocess uploaded image for NASNet Large model."""
    img = image.load_img(image_path, target_size=(331, 331))  # Resize image
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

def predict_image(request):
    """Handle image upload and prediction."""
    if request.method == 'POST' and request.FILES['image']:
        image_file = request.FILES['image']
        image_path = default_storage.save('uploads/' + image_file.name, image_file)

        # Process and predict
        processed_image = preprocess_image(default_storage.path(image_path))
        predictions = model.predict(processed_image)[0]  # Extract the first prediction

        # Get predicted class index and confidence score
        predicted_class_index = np.argmax(predictions)
        predicted_label = class_labels[predicted_class_index]
        confidence_score = predictions[predicted_class_index] * 100  # Convert to percentage

        # Prepare class-wise confidence scores
        class_confidences = {class_labels[i]: round(predictions[i] * 100, 2) for i in range(len(predictions))}

        return render(request, 'predict.html', {
            'uploaded_image': default_storage.url(image_path),
            'prediction': predicted_label,
            'confidence_score': round(confidence_score, 2),
            'class_confidences': class_confidences
        })

    return render(request, 'predict.html')

def homepage(request):
    return render(request, 'index.html')
