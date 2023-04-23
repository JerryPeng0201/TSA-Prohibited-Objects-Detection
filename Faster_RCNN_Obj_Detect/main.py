from predict import *


if __name__ == '__main__':
    img_path = 'examples/pred_1.png'
    model_path = 'Trained-Models/pre-train/epoch_80.pth'
    predict_bbox(model_path, img_path)