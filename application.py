import glob
import os
import json
from typing import Dict, Any

from flask import Flask, request
from werkzeug.utils import secure_filename

import torchvision

from utils.predictor import predict

convert_to_pil = torchvision.transforms.ToPILImage()

UPLOAD_FOLDER = 'static/uploads/'

application = Flask(__name__)
application.secret_key = "secret key"
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
application.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@application.route('/v1/annotate', methods=['POST'])
def annotate_image() -> Dict[str, Any]:
    try:
        # Clear old uploaded files
        directory = os.getcwd()
        files = glob.glob(f'{directory}/static/uploads/*')
        for f in files:
            os.remove(f)

        if 'file' not in request.files:
            return json.dumps({'error': 'No file sent'}), 500

        file = request.files['file']

        if file.filename == '':
            return json.dumps({'error': 'No image selected for uploading'}), 500

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_image_path = os.path.join(application.config['UPLOAD_FOLDER'], filename)
            output_image_path = os.path.join(application.config['UPLOAD_FOLDER'], 'annotated_' + filename)

            file.save(input_image_path)

            predicted_picture_output = predict(input_image_path)
            predicted_picture = predicted_picture_output['image_with_boxes']
            bounding_box_coords = predicted_picture_output['box_coordinates']

            pil_image = convert_to_pil(predicted_picture)

            pil_image.save(output_image_path)

            return json.dumps({'input_image': input_image_path,
                               'annotated_image': output_image_path,
                               'bounding_box_coords': bounding_box_coords})

        return json.dumps({'error': 'Allowed image types are -> png, jpg, jpeg'})

    except Exception as e:
        return json.dumps({'error': f'An error occurred: {str(e)}'}), 500


if __name__ == "__main__":
    application.run()
