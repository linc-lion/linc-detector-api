import json
import time

import torch
import torchvision
from linc.detector.helper.utils import fetch_boxes_coordinates
from linc.detector.models import detection

from utils.logger_factory import LoggerFactory

logger = LoggerFactory.create_logger(service_name='linc-detector-api', logger_name=__name__)


class Predictor:
    def __init__(self):
        self.device = 'cpu'
        self.draw_confidence_threshold = 0.5
        self.checkpoint = self.load_checkpoint()
        self.model = self.build_model(self.checkpoint)
        self.to_tensor = torchvision.transforms.ToTensor()

    def load_checkpoint(self):
        # Load the model and checkpoint during initialization
        with open('config/artifacts.json', 'r') as file:
            data = json.load(file)

        model_filename = f"artifacts/{data['name']}/{data['version']}/model.pth"
        model_checkpoint = torch.load(
            model_filename,
            map_location=self.device
        )
        return model_checkpoint

    def build_model(self, checkpoint):
        loaded_model = detection.fasterrcnn_resnet50_fpn(
            num_classes=len(checkpoint['label_names']) + 1, pretrained_backbone=False
        )

        loaded_model.to(self.device)
        loaded_model.load_state_dict(checkpoint['model'])
        loaded_model.eval()
        return loaded_model

    def predict(self, image):
        tensor_image = self.to_tensor(image).to(self.device)

        label_names = self.checkpoint['label_names']

        logger.info('Running image through model...')
        tic = time.time()
        outputs = self.model([tensor_image])
        toc = time.time()
        logger.info(f'Done in {toc - tic:.2f} seconds!')

        scores = outputs[0]['scores']
        top_scores_filter = scores > self.draw_confidence_threshold
        top_scores = scores[top_scores_filter]
        top_boxes = outputs[0]['boxes'][top_scores_filter]
        top_labels = outputs[0]['labels'][top_scores_filter]

        if len(top_scores) > 0:
            logger.info(f'Number of detected objects: {len(top_scores)}')
            box_coordinates = fetch_boxes_coordinates(tensor_image, top_boxes, top_labels, label_names)
            return {"box_coordinates": box_coordinates}
        else:
            return {"error": "No objects detected with confidence above the threshold."}
