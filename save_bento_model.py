import bentoml
import boto3
import os
from linc.detector.models import detection
import torch
from linc.detector.models.detection import FasterRCNN


MODEL_FILENAME = 'model.pth'
device = 'cpu'

if __name__ == "__main__":

    def download_model():
        session = boto3.Session()
        s3 = session.resource('s3')
        BUCKET_NAME = 'linc-model-artifact'
        my_bucket = s3.Bucket(BUCKET_NAME)
        KEY = 'linc-detector/20221002/model.pth'
        my_bucket.download_file(KEY, 'model.pth')


    def load_check_point():
        if not os.path.exists(MODEL_FILENAME):
            download_model()
        print('Loading checkpoint from hardrive... ', end='', flush=True)
        model_checkpoint = torch.load(MODEL_FILENAME, map_location=device)

        return model_checkpoint


    def build_model(model_checkpoint):
        print('Building model and loading checkpoint into it... ', end='', flush=True)
        loaded_model = detection.fasterrcnn_resnet50_fpn(
            num_classes=len(model_checkpoint['label_names']) + 1, pretrained_backbone=False
        )

        loaded_model.to(device)

        loaded_model.load_state_dict(checkpoint['model'])
        loaded_model.eval()

        return loaded_model


    checkpoint = load_check_point()
    model = build_model(checkpoint)

    def save_model(loaded_model):

        bentoml.pytorch.save_model(
            "habiba",   # Model name in the local Model Store
            loaded_model,  # Model instance being saved
            labels={    # User-defined labels for managing models in BentoCloud
                "owner": "lion_guardians",
                "stage": "dev",
            },
            metadata={},
            custom_objects={}
        )


    save_model(model)
