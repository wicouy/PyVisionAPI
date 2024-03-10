from flask import Flask, request, jsonify
from google.cloud import vision
import os
import json

# Ruta absoluta al directorio donde reside este script
basedir = os.path.abspath(os.path.dirname(__file__))

# Cargar configuración desde el archivo en el directorio 'seguridad'
config_path = os.path.join(basedir, 'seguridad', 'config.json')
with open(config_path) as config_file:
    config = json.load(config_file)

# Establecer la variable de entorno para las credenciales de Google Cloud
google_credentials_path = os.path.join(basedir, 'seguridad', config['google_cloud_credentials'])
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_credentials_path

# Inicializar la aplicación Flask y el cliente de Google Cloud Vision
app = Flask(__name__)
client = vision.ImageAnnotatorClient()

@app.route('/detect-landmarks', methods=['POST'])
def detect_landmarks():
    # Verificar si la imagen está en la petición
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found in the request'}), 400

    # Leer la imagen de la petición
    image_file = request.files['image']
    content = image_file.read()
    image = vision.Image(content=content)

    # Llamada a Google Cloud Vision API para detectar puntos de referencia
    try:
        response = client.landmark_detection(image=image)
        landmarks = response.landmark_annotations

        # Preparar y enviar la respuesta
        landmarks_info = [{'name': landmark.description, 'score': landmark.score} for landmark in landmarks]
        return jsonify(landmarks_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ejecutar la aplicación Flask
if __name__ == '__main__':
    app.run(debug=config['debug'], host=config['host'], port=config['port'])
