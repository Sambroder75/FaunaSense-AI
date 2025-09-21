import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))

from flask import Flask, request, jsonify, render_template
from models.model import predict_animal, get_animal_info  # Import functions
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure 'uploads' folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/identify', methods=['POST'])
def identify():
    # Check for file in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']

    # Check for empty filename
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save and process valid file types
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Predict the animal in the image
            animal_name = predict_animal(file_path)  # Should return a single name

            if not animal_name:
                return jsonify({'error': 'Prediction failed, no animal detected.'}), 500

            # Retrieve animal information
            animal_details = get_animal_info(animal_name)
            if not animal_details:
                return jsonify({'error': f'No information found for animal: {animal_name}'}), 404

            # Return the result as JSON
            result = {
                "Animal": animal_name,
                "Classification": animal_details.get('classification', 'Unknown'),
                "Venomous": animal_details.get('venomous', 'Unknown'),
                "Dangerous": animal_details.get('dangerous', 'Unknown'),
                "How to get rid of": animal_details.get('how_to_get_rid_of', 'Not specified')
            }
            return jsonify(result)

        except Exception as e:
            # Handle unexpected errors
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True)