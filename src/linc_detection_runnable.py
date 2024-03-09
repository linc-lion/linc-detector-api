import bentoml

from adapter.predictor import Predictor


class LincDetectionRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.predictor = Predictor()

    @bentoml.Runnable.method(batchable=False)
    def inference(self, image):
        return self.predictor.predict(image)
