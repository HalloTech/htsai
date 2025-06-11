import os
from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from datetime import datetime
import traceback
import torch
# Import the model predictor
from inference.tryon import LeffaPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] using device: {device} ")

app = Flask(__name__)

# Folder configs
UPLOAD_FOLDER_USER = 'static/uploads/user'
UPLOAD_FOLDER_PRODUCT = 'static/uploads/product'
RESULT_FOLDER = 'static/results'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER_USER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_PRODUCT, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER_USER'] = UPLOAD_FOLDER_USER
app.config['UPLOAD_FOLDER_PRODUCT'] = UPLOAD_FOLDER_PRODUCT
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Initialize predictor once
predictor = LeffaPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    user_img = request.files['user_image']
    product_img = request.files['product_image']

    if user_img and product_img:
        user_filename = secure_filename(user_img.filename)
        product_filename = secure_filename(product_img.filename)

        user_path = os.path.join(UPLOAD_FOLDER_USER, user_filename)
        product_path = os.path.join(UPLOAD_FOLDER_PRODUCT, product_filename)

        user_img.save(user_path)
        product_img.save(product_path)

        user_image = Image.open(user_path)
        product_image = Image.open(product_path)

        h = min(user_image.height, product_image.height)
        user_image = user_image.resize((int(user_image.width * h / user_image.height), h))
        product_image = product_image.resize((int(product_image.width * h / product_image.height), h))

        result_img = predictor.run_tryon(user_path, product_path)
        result_filename = f"result_{user_filename.split('.')[0]}_{product_filename.split('.')[0]}.png"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        result_img.save(result_path)


        user_image_path = os.path.relpath(user_path, 'static')
        product_image_path = os.path.relpath(product_path, 'static')
        result_image_path = os.path.relpath(result_path, 'static')

        return render_template('index.html',
                               user_image_path=user_image_path,
                               product_image_path=product_image_path,
                               result_image_path=result_image_path)

    return "Upload failed", 400

@app.route("/tryon", methods=["POST"])
def tryon_api():
    try:
        user_file = request.files.get("user")
        cloth_file = request.files.get("cloth")

        if not user_file or not cloth_file:
            return jsonify({"error": "Missing user or cloth image"}), 400

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        user_path = os.path.join(UPLOAD_FOLDER_USER, f"user_{timestamp}.jpg")
        cloth_path = os.path.join(UPLOAD_FOLDER_PRODUCT, f"cloth_{timestamp}.jpg")
        result_path = os.path.join(RESULT_FOLDER, f"result_{timestamp}.png")

        user_file.save(user_path)
        cloth_file.save(cloth_path)

        # Run virtual try-on AI
        result_img = predictor.run_tryon(user_path, cloth_path)
        result_img.save(result_path)

        return jsonify({"result": os.path.relpath(result_path, 'static')})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(RESULT_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
