import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image


def predict_bbox(model_path, img_path):
    # Load the trained Faster R-CNN model
    print("Loading the pretrained model ...")
    model = torch.load(model_path)
    model.eval()

    # Load and preprocess the image
    image = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0) 

    print("Predicting the bounding boxes ...")
    # Run the model on the image
    with torch.no_grad():
        outputs = model(image_tensor)

    # Extract the predicted bounding boxes and scores
    boxes = outputs[0]['boxes']
    scores = outputs[0]['scores']
    labels = outputs[0]['labels']

    # Set a score threshold to filter out low-confidence predictions
    score_threshold = 0.5

    # Filter out predictions with scores below the threshold
    keep_indices = torch.nonzero(scores > score_threshold).squeeze(1)
    filtered_boxes = boxes[keep_indices]
    filtered_labels = labels[keep_indices]

    # Convert the image to a numpy array
    image_np = np.array(image)

    # Draw the predicted bounding boxes on the image
    for box, label in zip(filtered_boxes, filtered_labels):
        x1, y1, x2, y2 = box.numpy().astype(int)
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_np, str(label.item()), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the result
    cv2.imshow('Result', image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()