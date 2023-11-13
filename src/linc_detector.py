from typing import Any, Dict

import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray
from numpy.typing import NDArray

bento_model = bentoml.pytorch.get("habiba:latest")


class LincDetectionRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        # load the model instance
        self.classifier = bentoml.pytorch.load_model(bento_model)

    @bentoml.Runnable.method(batchable=False)
    def is_spam(self, input_data: NDArray[Any]) -> NDArray[Any]:

        return self.classifier(input_data)


linc_detector_runner = bentoml.Runner(LincDetectionRunnable, models=[bento_model])
svc = bentoml.Service("linc_detector", runners=[linc_detector_runner])
print(svc)


@svc.api(input=NumpyNdarray(), output=JSON(), route='/v1/annotate')
def analysis(input_text: NDArray[Any]) -> Dict[str, Any]:
    return {"res": linc_detector_runner.is_spam.run(input_text)}


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