from io import BytesIO
from typing import Any, Dict, Union
import loggerFactory
import glob
import os
import json
import bentoml
from bentoml.io import JSON, Image, Multipart
from PIL.Image import Image as PILImage
from flask import jsonify
from numpy.typing import NDArray
from werkzeug.utils import secure_filename
from utils.predictor import predict
import numpy as np
import torchvision
import base64

bento_model = bentoml.pytorch.get("habiba:latest")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

logger = loggerFactory.StreamOnlyLogger(rand_name=True)

img_json_output_spec = Multipart(input_image=Image(), bounding_box_coords=JSON())


class LincDetectionRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        # load the model instance
        self.detector = bentoml.pytorch.load_model(bento_model)

    @bentoml.Runnable.method(batchable=False)
    def has_lion(self, input_data: NDArray[Any]) -> NDArray[Any]:

        return self.detector[(input_data)]


linc_detector_runner = bentoml.Runner(LincDetectionRunnable, models=[bento_model])
svc = bentoml.Service("linc_detector", runners=[linc_detector_runner])
print(svc)


@svc.api(input=Image(), output=img_json_output_spec, route='/v1/annotate')
def detect(image: PILImage, ctx: bentoml.Context) -> Union[Dict[str, str], Dict[str, str], Dict[str, str], str]:
    try:
        clear_old_files()

        headers = ctx.request.headers

        # Check if 'content-type' key is present in headers
        content_type = headers.get('content-type')

        if content_type is None:
            ctx.response.status_code = 405
            return {'error': 'Content-Type header is missing'}

        if not image:
            return {'error': 'No file sent'}

        query_params = ctx.request.query_params
        if 'vert_size' in ctx.request.query_params:
            vert_size = int(ctx.request.query_params.get('vert_size'))
        else:
            vert_size = 500
        if not allowed_file(image.format):
            ctx.response.status_code = 500
            allowed_types_error = {'error': 'Allowed image types are -> png, jpg, jpeg'}
            return allowed_types_error
        output = process_uploaded_file(image, vert_size)

        image_with_boxes, bounding_box_coords = output

        return {'input_image': image, 'bounding_box_coords': bounding_box_coords}

    except Exception:
        ctx.response.status_code = 500
        logger.error("Exception in predict method!")


def clear_old_files():
    directory = os.getcwd()
    files = glob.glob(f'{directory}/static/uploads/*')
    for f in files:
        os.remove(f)


def process_uploaded_file(file, vert_size):
    file.save("input_image_path", 'png')

    predicted_picture_output = predict("input_image_path", vert_size=vert_size)
    bounding_box_coords = predicted_picture_output['box_coordinates']
    image_with_boxes = predicted_picture_output["image_with_boxes"]
    return image_with_boxes, bounding_box_coords


def allowed_file(filename):
    print(filename)
    print("filename in allowed files")
    return filename.lower() in ALLOWED_EXTENSIONS
