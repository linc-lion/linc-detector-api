from typing import Dict, Union
import os
import glob
from PIL import Image as PILImage
import bentoml
from bentoml.io import JSON, Image
from linc_detection_runnable import LincDetectionRunnable
import loggerFactory
from domain.linc_detection_response import LincDetectionResponse, BoundingBoxCoords

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

logger = loggerFactory.StreamOnlyLogger(rand_name=True)
linc_detector_runner = bentoml.Runner(LincDetectionRunnable)
svc = bentoml.Service("linc_detection", runners=[linc_detector_runner])


@svc.api(input=Image(), output=JSON(), route='/v1/annotate')
def detect_v1(image: PILImage, ctx: bentoml.Context) -> LincDetectionResponse:
    # try:
    clear_old_files()

    content_type = ctx.request.headers.get('content-type')
    if content_type is None:
        ctx.response.status_code = 405
        return {'error': 'Content-Type header is missing'}

    if not image:
        return {'error': 'No file sent'}

    vert_size = int(ctx.request.query_params.get('vert_size', 500))
    bounding_box_coords = process_uploaded_file(image, vert_size, ctx)
    bounding_box_coords_instance = BoundingBoxCoords(**{'bounding_boxes': bounding_box_coords})
    return LincDetectionResponse(bounding_box_coords=bounding_box_coords_instance)

    # except Exception:
    #     ctx.response.status_code = 500
    #     logger.error("Exception in predict method!")


def clear_old_files():
    directory = os.getcwd()
    files = glob.glob(f'{directory}/static/uploads/*')
    for f in files:
        os.remove(f)


def process_uploaded_file(file, vert_size, ctx):
    if not allowed_file(file.format):
        ctx.response.status_code = 500
        allowed_types_error = {'error': 'Allowed image types are -> png, jpg, jpeg'}
        return allowed_types_error

    file.save("input_image_path", 'png')

    # Call predict method to get predictions
    prediction_result = linc_detector_runner.inference.run("input_image_path", vert_size)

    # Assuming prediction_result is a dictionary containing the bounding box coordinates
    bounding_box_coords = prediction_result.get("box_coordinates", None)
    return bounding_box_coords


def allowed_file(filename):
    return filename.lower() in ALLOWED_EXTENSIONS
