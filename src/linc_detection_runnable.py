# original_file.py
import bentoml
from adapter import predictor

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

draw_confidence_threshold = 0.5


class LincDetectionRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        pass

    @bentoml.Runnable.method(batchable=False)
    def inference(self, image_path, vert_size):
        return predictor.predict(image_path, vert_size)
