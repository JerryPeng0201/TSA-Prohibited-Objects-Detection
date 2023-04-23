import os
import cv2
import numpy as np
from skimage.feature import hog
from pycocotools.coco import COCO


class PIDrayDataset:
    def __init__(self, annotation_path, img_dir, resize_shape=(64,64), debug_mode=False):
        self.img_dir = img_dir
        self.coco = COCO(annotation_path)
        self.ids = self.coco.getImgIds()
        self.images = self.coco.loadImgs(self.ids)
        self.resize_shape = resize_shape
        self.debug_mode = debug_mode
    
    
    def parse_data(self):
        hog_features = []
        labels = []

        for i, image in enumerate(self.images):
            if i % 1000 == 0:
                print("Loading images from No.{} to No.{}".format(i+1, i+1000))

            # Read each image
            img_path = os.path.join(self.img_dir, image['file_name'])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue
            
            # Load annotations
            ann_ids = self.coco.getAnnIds(imgIds=image['id'])
            anns = self.coco.loadAnns(ann_ids)

            # pase annotations data
            for i, ann in enumerate(anns):
                bbox = ann['bbox']
                x, y, w, h = [int(coord) for coord in bbox]
                class_id = ann['category_id']
                # Label the roi on the image
                roi = img[y:y+h, x:x+w]
                try:
                    resized_img = cv2.resize(roi, self.resize_shape)
                except cv2.error as e:
                    print(f"Error resizing image (ID: {i}): {e}")
                    continue
                #resized_img = cv2.resize(roi, self.resize_shape)

                # HOG feature extraction
                fd = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)

                # Update the output
                hog_features.append(fd)
                labels.append(class_id)

            # Debugging?
            if self.debug_mode and i == 999:
                print("You are using debugging mode.")
                break
        
        return np.array(hog_features), np.array(labels)
            