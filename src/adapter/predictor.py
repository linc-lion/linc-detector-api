import time
from linc.detector.helper.utils import fetch_boxes_coordinates
from linc.detector.models import detection
import torch
import torchvision
from utils.logger_factory import LoggerFactory

draw_confidence_threshold = 0.5

to_tensor = torchvision.transforms.ToTensor()
device = 'cpu'
BUCKET_NAME = 'linc-model-artifact'

logger = LoggerFactory.create_logger(service_name='linc-detector-api', logger_name=__name__)


def load_checkpoint(model_name, model_version):
    model_filename = f'../artifacts/{model_name}/{model_version}/model.pth'
    model_checkpoint = torch.load(
        model_filename,
        map_location=device
    )
    return model_checkpoint


def build_model(checkpoint):
    loaded_model = detection.fasterrcnn_resnet50_fpn(
        num_classes=len(checkpoint['label_names']) + 1, pretrained_backbone=False
    )

    loaded_model.to(device)
    loaded_model.load_state_dict(checkpoint['model'])
    loaded_model.eval()
    return loaded_model


def predict(model, checkpoint, image):

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
        box_coordinates = fetch_boxes_coordinates(tensor_image, top_boxes, top_labels, label_names)
        return {"box_coordinates": box_coordinates}
    else:
        return {"error": "No objects detected with confidence above the threshold."}
