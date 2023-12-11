import os
import glob
from PIL import Image as PILImage
import bentoml
from bentoml.io import JSON, Image
from linc_detection_runnable import LincDetectionRunnable
import loggerFactory
from domain.linc_detection_response import LincDetectionResponse

logger = loggerFactory.StreamOnlyLogger(rand_name=True)
linc_detector_runner = bentoml.Runner(LincDetectionRunnable)
svc = bentoml.Service("linc_detection", runners=[linc_detector_runner])


@svc.api(input=Image(), output=JSON(), route='/v1/annotate')
def detect_v1(image: PILImage, ctx: bentoml.Context) -> LincDetectionResponse:
    try:

        clear_old_files()

        bounding_box_coords = process_uploaded_file(image, ctx)
        return LincDetectionResponse(bounding_box_coords=bounding_box_coords)

    except Exception:
        ctx.response.status_code = 500
        return {"error": "No lion detected in image"}


def clear_old_files():
    directory = os.getcwd()
    files = glob.glob(f'{directory}/static/uploads/*')
    for f in files:
        os.remove(f)


def process_uploaded_file(file, ctx):

    file.save("input_image_path", 'png')

    # Call predict method to get predictions
    prediction_result = linc_detector_runner.inference.run("input_image_path")

    # Assuming prediction_result is a dictionary containing the bounding box coordinates
    bounding_box_coords = prediction_result.get("box_coordinates", None)
    return bounding_box_coords
