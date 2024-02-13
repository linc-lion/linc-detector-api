import json

import bentoml
from adapter import predictor

draw_confidence_threshold = 0.5


class LincDetectionRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.model, self.checkpoint = self._init_model_and_checkpoint()

    @staticmethod
    def _init_model_and_checkpoint():
        # Load the model and checkpoint during initialization
        with open('config/artifacts.json', 'r') as file:
            data = json.load(file)
        checkpoint = predictor.load_checkpoint(data['name'], data['version'])
        model = predictor.build_model(checkpoint)
        return model, checkpoint

    @bentoml.Runnable.method(batchable=False)
    def inference(self, image):
        return predictor.predict(self.model, self.checkpoint, image)
