from PIL import Image as PILImage
import bentoml
from bentoml.io import JSON, Image
from linc_detection_runnable import LincDetectionRunnable
from domain.linc_detection_response import LincDetectionResponse

linc_detector_runner = bentoml.Runner(LincDetectionRunnable)
svc = bentoml.Service("linc_detection", runners=[linc_detector_runner])


@svc.api(input=Image(), output=JSON(), route='/v1/annotate')
async def predict(image: PILImage, ctx: bentoml.Context) -> LincDetectionResponse:
    try:

        # Call predict method to get predictions
        prediction_result = await linc_detector_runner.inference.async_run(image)

        # Assuming prediction_result is a dictionary containing the bounding box coordinates
        bounding_box_coords = prediction_result.get("box_coordinates", None)

        return LincDetectionResponse(bounding_box_coords=bounding_box_coords)

    except Exception:

        ctx.response.status_code = 500
        # Return LincDetectionResponse with an error message in the error case
        return LincDetectionResponse(bounding_box_coords=None, error_message="No lion detected in image")

