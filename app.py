from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import requests
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = load_model('nails_classifier_model.keras')


# Class indices (replace with your actual indices)
class_indices = {
    0: 'Acral_Lentiginous_Melanoma',
    1: 'blue_finger',
    2: 'clubbing',
    3: 'Healthy_Nail',
    4: 'Onychogryphosis',
    5: 'pitting',
}

@app.route("/")
def helloworld():
    return "Hello World!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image URL from request
        data = request.get_json()
        image_url = data.get('image_url')
        if not image_url:
            return jsonify({"error": "No image URL provided"}), 400

        # Download and preprocess the image
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image = image.resize((224, 224))  # Resize to match model input
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        image_array = image_array / 255.0  # Normalize

        # Predict using the model
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions)
        confidence_scores = {class_indices[k]: float(predictions[0][k] * 100) for k in range(len(class_indices))}

        # Return prediction
        return jsonify({
            "predicted_class": class_indices[predicted_class],
            "confidence_scores": confidence_scores
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
