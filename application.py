import glob
import os
from typing import Union, Tuple

import torchvision
from flask import Flask, request, jsonify, Response
from flask_httpauth import HTTPTokenAuth
from werkzeug.utils import secure_filename

from utils.predictor import predict

convert_to_pil = torchvision.transforms.ToPILImage()

UPLOAD_FOLDER = 'static/uploads/'

application = Flask(__name__)
auth = HTTPTokenAuth(scheme='Bearer')

API_KEY = '1e620008-745c-4e84-be74-81042ab71b1e'

application.secret_key = "secret key"
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
application.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024





@application.route('/v1/annotate', methods=['POST'])
@auth.login_required
def annotate_image() -> Union[Tuple[Response, int], Response]:
    try:
        vert_size = int(request.args.get('vert_size', 500))

        # Clear old uploaded files
        clear_old_files()

        if 'file' not in request.files:
            return jsonify({'error': 'No file sent'}), 500

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No image selected for uploading'}), 500

        input_image_path, bounding_box_coords = process_uploaded_file(file, vert_size)

        return jsonify({'input_image': input_image_path,
                        'bounding_box_coords': bounding_box_coords})

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


def clear_old_files():
    directory = os.getcwd()
    files = glob.glob(f'{directory}/static/uploads/*')
    for f in files:
        os.remove(f)


def process_uploaded_file(file, vert_size):
    if not allowed_file(file.filename):
        return jsonify({'error': 'Allowed image types are -> png, jpg, jpeg'}), 500

    filename = secure_filename(file.filename)
    input_image_path = os.path.join(application.config['UPLOAD_FOLDER'], filename)
    output_image_path = os.path.join(application.config['UPLOAD_FOLDER'], 'annotated_' + filename)

    file.save(input_image_path)

    predicted_picture_output = predict(input_image_path, vert_size=vert_size)
    bounding_box_coords = predicted_picture_output['box_coordinates']

    return input_image_path, bounding_box_coords


@auth.verify_token
def verify_token(token):
    print("Received token:", token)
    return token == API_KEY


if __name__ == "__main__":
    application.run()
