# predictor.py

import os
import time
import PIL
import boto3
from linc.detector.helper.utils import draw_boxes, fetch_boxes_coordinates
from linc.detector.models import detection
import torch
import torchvision


draw_confidence_threshold = 0.5

to_tensor = torchvision.transforms.ToTensor()
device = 'cpu'
BUCKET_NAME = 'linc-model-artifact'
KEY = 'linc-detector/20221002/model.pth'
model_filename = 'model.pth'
device = 'cpu'


def load_checkpoint():
    if not os.path.exists(model_filename):
        download_model()
    print('Loading checkpoint from hard drive... ', end='', flush=True)
    model_checkpoint = torch.load(model_filename, map_location=device)
    return model_checkpoint


def download_model():
    session = boto3.Session()
    s3 = session.resource('s3')
    my_bucket = s3.Bucket(BUCKET_NAME)
    my_bucket.download_file(KEY, 'model.pth')


def build_model(checkpoint):
    print('Building model and loading checkpoint into it... ', end='', flush=True)
    loaded_model = detection.fasterrcnn_resnet50_fpn(
        num_classes=len(checkpoint['label_names']) + 1, pretrained_backbone=False
    )

    loaded_model.to(device)
    loaded_model.load_state_dict(checkpoint['model'])
    loaded_model.eval()
    return loaded_model


def predict(image_path, vert_size):
    print(f"Running inference on {device} device")
    print(f"image_path {image_path}")
    print('Loading image... ', end='', flush=True)
    loaded_image = PIL.Image.open(image_path)
    width, height = loaded_image.size
    print(f'Input image_width: {width}, image_height: {height}')
    image = to_tensor(loaded_image).to(device)
    print('Done.')

    checkpoint = load_checkpoint()
    label_names = checkpoint['label_names']

    print('Running image through model... ', end='', flush=True)
    tic = time.time()
    model = build_model(checkpoint)
    outputs = model([image])
    toc = time.time()
    print(f'Done in {toc - tic:.2f} seconds!')

    scores = outputs[0]['scores']
    top_scores_filter = scores > draw_confidence_threshold
    top_scores = scores[top_scores_filter]
    top_boxes = outputs[0]['boxes'][top_scores_filter]
    top_labels = outputs[0]['labels'][top_scores_filter]

    if len(top_scores) > 0:
        image_with_boxes = draw_boxes(
            image.cpu(), top_boxes, top_labels.cpu(), label_names, scores, vert_size=vert_size
        )

        width, height = loaded_image.size
        print(f'Predicted image_width: {width}, image_height: {height}')
        box_coordinates = fetch_boxes_coordinates(image, top_boxes, top_labels, label_names)
        return {"image_with_boxes": image_with_boxes, "box_coordinates": box_coordinates}
    else:
        print("The model didn't find any object it feels confident enough to show")
        return False
