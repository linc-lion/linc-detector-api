import glob
import os
import uuid
from typing import Union, Tuple

import torchvision
from flask import Flask, request, jsonify, Response
from flask_httpauth import HTTPTokenAuth
from werkzeug.utils import secure_filename

from utils.predictor import predict

convert_to_pil = torchvision.transforms.ToPILImage()

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
auth = HTTPTokenAuth(scheme='Bearer')

API_KEY = '1e620008-745c-4e84-be74-81042ab71b1e'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/v1/annotate', methods=['POST'])
@auth.login_required  # Require authentication to access this endpoint
def annotate_image() -> Union[Tuple[Response, int], Response]:
    try:
        vert_size = int(request.args.get('vert_size', 500))  # Default value is 500

        # Clear old uploaded files
        directory = os.getcwd()
        files = glob.glob(f'{directory}/static/uploads/*')
        for f in files:
            os.remove(f)

        if 'file' not in request.files:
            return jsonify({'error': 'No file sent'}), 500

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No image selected for uploading'}), 500

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_' + filename)

            file.save(input_image_path)

            predicted_picture_output = predict(input_image_path, vert_size=vert_size)
            # predicted_picture = predicted_picture_output['image_with_boxes']
            bounding_box_coords = predicted_picture_output['box_coordinates']

            # pil_image = convert_to_pil(predicted_picture)

            # pil_image.save(output_image_path)

            return jsonify({'input_image': input_image_path,
                            'bounding_box_coords': bounding_box_coords})

        return jsonify({'error': 'Allowed image types are -> png, jpg, jpeg'})

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@auth.verify_token
def verify_token(token):
    print("Received token:", token)
    habiba_api_key = str(uuid.uuid4())
    print(habiba_api_key)
    return token == API_KEY


if __name__ == "__main__":
    app.run()
