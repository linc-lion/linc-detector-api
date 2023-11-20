from typing import Dict, Union
import loggerFactory
import glob
import os
import bentoml
from bentoml.io import JSON, Image
from PIL.Image import Image as PILImage
from linc_detection_runnable import LincDetectionRunnable

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

logger = loggerFactory.StreamOnlyLogger(rand_name=True)


linc_detector_runner = bentoml.Runner(LincDetectionRunnable)
svc = bentoml.Service("linc_detection", runners=[linc_detector_runner])


@svc.api(input=Image(), output=JSON(), route='/v1/annotate')
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
        if 'vert_size' in query_params:
            vert_size = int(query_params.get('vert_size'))
        else:
            vert_size = 500
        bounding_box_coords = process_uploaded_file(image, vert_size, ctx)

        return {'bounding_box_coords': bounding_box_coords}

    except Exception:
        ctx.response.status_code = 500
        logger.error("Exception in predict method!")


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
    prediction_result = linc_detector_runner.predict.run("input_image_path", vert_size)

    # Assuming prediction_result is a dictionary containing the bounding box coordinates
    bounding_box_coords = prediction_result.get("box_coordinates", None)
    return bounding_box_coords


def allowed_file(filename):
    print(filename)
    print("filename in allowed files")
    return filename.lower() in ALLOWED_EXTENSIONS
