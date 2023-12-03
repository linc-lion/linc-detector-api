# predictor.py

import os
import time
import PIL
import boto3
from linc.detector.helper.utils import draw_boxes, fetch_boxes_coordinates
from linc.detector.models import detection
import torch
import torchvision
from utils.logger_factory import LoggerFactory

draw_confidence_threshold = 0.5

to_tensor = torchvision.transforms.ToTensor()
device = 'cpu'
BUCKET_NAME = 'linc-model-artifact'
KEY = 'linc-detector/20221002/model.pth'
model_filename = 'model.pth'

logger = LoggerFactory.create_logger(service_name='linc-detector-api', logger_name=__name__)


def load_checkpoint():
    if not os.path.exists(model_filename):
        download_model()
    model_checkpoint = torch.load(model_filename, map_location=device)
    return model_checkpoint


def download_model():
    session = boto3.Session()
    s3 = session.resource('s3')
    bucket = s3.Bucket(BUCKET_NAME)
    bucket.download_file(KEY, 'model.pth')


def build_model(checkpoint):
    loaded_model = detection.fasterrcnn_resnet50_fpn(
        num_classes=len(checkpoint['label_names']) + 1, pretrained_backbone=False
    )

    loaded_model.to(device)
    loaded_model.load_state_dict(checkpoint['model'])
    loaded_model.eval()
    return loaded_model


def predict(model, checkpoint, image_path, vert_size):
    image = PIL.Image.open(image_path)

    tensor_image = to_tensor(image).to(device)

    label_names = checkpoint['label_names']

    logger.info('Running image through model...')
    tic = time.time()
    outputs = model([tensor_image])
    toc = time.time()
    logger.info(f'Done in {toc - tic:.2f} seconds!')

    scores = outputs[0]['scores']
    top_scores_filter = scores > draw_confidence_threshold
    top_scores = scores[top_scores_filter]
    top_boxes = outputs[0]['boxes'][top_scores_filter]
    top_labels = outputs[0]['labels'][top_scores_filter]

    if len(top_scores) > 0:
        logger.info(f'Number of detected objects: {len(top_scores)}')
        image_with_boxes = draw_boxes(
            tensor_image.cpu(), top_boxes, top_labels.cpu(), label_names, scores, vert_size=vert_size
        )
        box_coordinates = fetch_boxes_coordinates(tensor_image, top_boxes, top_labels, label_names)
        return {"image_with_boxes": image_with_boxes, "box_coordinates": box_coordinates}
    else:
        logger.info("No objects detected with confidence above the threshold.")
        return False
