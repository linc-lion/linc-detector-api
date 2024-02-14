import os

from PIL import Image as PILImage
import bentoml
from bentoml.io import JSON, Image
from linc_detection_runnable import LincDetectionRunnable
from domain.linc_detection_response import LincDetectionResponse
from utils.logger_factory import LoggerFactory

logger = LoggerFactory.create_logger(service_name='linc-detector-api', logger_name=__name__)

linc_detector_runner = bentoml.Runner(LincDetectionRunnable)
svc = bentoml.Service("linc_detection", runners=[linc_detector_runner])

bearer_token = os.environ.get('BEARER_TOKEN')

logger.info(f"bearer_token: {bearer_token}")


def authenticate_bearer_token(request):
    auth_header = request.headers.get('Authorization')
    if auth_header:
        token_type, _, token = auth_header.partition(' ')
        logger.info(f"token: {token}")
        if token_type.lower() == 'bearer':
            if bearer_token and token == bearer_token:
                return True
    return False


@svc.api(input=Image(), output=JSON(), route='/v1/annotate')
async def predict(image: PILImage, ctx: bentoml.Context) -> LincDetectionResponse:
    try:

        is_authenticated = authenticate_bearer_token(ctx.request)
        if not is_authenticated:
            ctx.response.status_code = 401
            return LincDetectionResponse(bounding_box_coords=None, error_message="Unauthorized")

        # Call predict method to get predictions
        prediction_result = await linc_detector_runner.inference.async_run(image)

        # Assuming prediction_result is a dictionary containing the bounding box coordinates
        bounding_box_coords = prediction_result.get("box_coordinates", None)

        return LincDetectionResponse(bounding_box_coords=bounding_box_coords)

    except Exception:

        ctx.response.status_code = 500
        # Return LincDetectionResponse with an error message in the error case
        return LincDetectionResponse(bounding_box_coords=None, error_message="No lion detected in image")
