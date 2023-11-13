from io import BytesIO
from typing import Any, Dict
import glob
import os
import bentoml
from bentoml.io import JSON, Image
from PIL.Image import Image as PILImage
from numpy.typing import NDArray
from werkzeug.utils import secure_filename
from utils.predictor import predict

bento_model = bentoml.pytorch.get("habiba:latest")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


class LincDetectionRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        # load the model instance
        self.classifier = bentoml.pytorch.load_model(bento_model)

    @bentoml.Runnable.method(batchable=False)
    def is_spam(self, input_data: NDArray[Any]) -> NDArray[Any]:

        return self.classifier[(input_data)]


linc_detector_runner = bentoml.Runner(LincDetectionRunnable, models=[bento_model])
svc = bentoml.Service("linc_detector", runners=[linc_detector_runner])
print(svc)


@svc.api(input=Image(), output=JSON(), route='/v1/annotate')
def detect(image: PILImage, ctx: bentoml.Context) -> Dict[str, Any]:
    headers = ctx.request.headers

    # Check if 'content-type' key is present in headers
    content_type = headers.get('content-type')

    if content_type is None:
        ctx.response.status_code = 405
        return {'error': 'Content-Type header is missing'}

    if not image:
        return {'error': 'No file sent'}

    clear_old_files()
    query_params = ctx.request.query_params
    if 'vert_size' in ctx.request.query_params:
        vert_size = int(ctx.request.query_params.get('vert_size'))
    else:
        vert_size = 500
    print(vert_size)
    print("vert_size")
    print(allowed_file(image.format))
    print("allowed_file(image.format)")
    if not allowed_file(image.format):
        ctx.response.status_code = 500
        allowed_types_error = {'error': 'Allowed image types are -> png, jpg, jpeg'}
        return allowed_types_error
    habiba = process_uploaded_file(image, vert_size, ctx)
    print(habiba)
    print("habiba")
    predicted_picture_output, bounding_box_coords = habiba

    print(query_params)
    print("query_params")

    return {'input_image': predicted_picture_output, 'bounding_box_coords': bounding_box_coords}


def clear_old_files():
    directory = os.getcwd()
    files = glob.glob(f'{directory}/static/uploads/*')
    for f in files:
        os.remove(f)


def process_uploaded_file(file, vert_size, ctx):
    # buffer = BytesIO()
    # input.save(file, format=input.format)

    # filename = 'habiba'
    # print(filename)
    # print("filename")
    # input_image_path = os.path.join('upload_folder/')
    # output_image_path = os.path.join('upload_folder/', 'annotated_')

    # print(input_image_path)
    print("input_image_path")
    print(file.format)
    file.save("input_image_path", file.format)

    predicted_picture_output = predict("input_image_path", vert_size=vert_size)
    bounding_box_coords = predicted_picture_output['box_coordinates']
    return predicted_picture_output, bounding_box_coords


def allowed_file(filename):
    print(filename)
    print("filename in allowed files")
    return filename.lower() in ALLOWED_EXTENSIONS

    return input_image_path, bounding_box_coords
# print(f"Running inference on {device} device")
#
# print('Loading image... ', end='', flush=True)
# loaded_image = PIL.Image.open(image_path)
# width, height = loaded_image.size
# print(f'Input image_width: {width}, image_height: {height}')
# image = to_tensor(loaded_image).to(device)
# print('Done.')
#
# label_names = checkpoint['label_names']
#
# print('Running image through model... ', end='', flush=True)
# tic = time.time()