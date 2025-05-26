from flask import Flask, render_template, request, send_file, jsonify
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(image_path, style):
    # Read the image
    color_image = cv2.imread(image_path)
    
    if style == "1":  # Smooth Painting
        cartoon_image = cv2.stylization(color_image, sigma_s=200, sigma_r=0.1)
    else:  # Bold Sketchy
        # Convert to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # Apply median blur
        blurred = cv2.medianBlur(gray, 7)
        # Detect edges using adaptive threshold
        edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        # Apply bilateral filter for smooth color reduction
        color = cv2.bilateralFilter(color_image, d=9, sigmaColor=200, sigmaSpace=200)
        # Combine edges with color image
        cartoon_image = cv2.bitwise_and(color, color, mask=edges)
    
    return cartoon_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    style = request.form.get('style', '1')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        try:
            # Process the image
            processed_image = process_image(filepath, style)
            
            # Save processed image
            output_filename = f"processed_{unique_filename}"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            cv2.imwrite(output_path, processed_image)
            
            return jsonify({
                'success': True,
                'processed_image': output_filename
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up original file
            os.remove(filepath)
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(app.config['UPLOAD_FOLDER'], filename),
        as_attachment=True
    )

if __name__ == '__main__':
    app.run(debug=True) 