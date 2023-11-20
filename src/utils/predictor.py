import torchvision

from linc.detector.helper.utils import draw_boxes, fetch_boxes_coordinates
import time
import PIL.Image
import torch

draw_confidence_threshold = 0.5

to_tensor = torchvision.transforms.ToTensor()

model_filename = 'model.pth'
device = 'cpu'

@torch.no_grad()
def predict(image_path, vert_size, checkpoint, model):
    print(f"Running inference on {device} device")

    print('Loading image... ', end='', flush=True)
    loaded_image = PIL.Image.open(image_path)
    width, height = loaded_image.size
    print(f'Input image_width: {width}, image_height: {height}')
    image = to_tensor(loaded_image).to(device)
    print('Done.')

    label_names = checkpoint['label_names']

    print('Running image through model... ', end='', flush=True)
    tic = time.time()
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
        print("The model didn't find any object it feels confident about enough to show")
        return False
