import os
import cv2
import json
import numpy as np
from skimage import exposure
from skimage.feature import hog

def hoggles(image, bbox):
    x, y, w, h = [int(coord) for coord in bbox]
    cropped_image = image[y:y+h, x:x+w]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute HOG features and visualization
    hog_features, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

    # Enhance the visualization
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # Normalize the HOG image to a range between 0 and 255
    hog_image_normalized = ((hog_image - hog_image.min()) * (255 / (hog_image.max() - hog_image.min()))).astype(np.uint8)

    # Convert the normalized grayscale HOG image to a 3-channel image
    hog_image_3_channel = cv2.cvtColor(hog_image_normalized, cv2.COLOR_GRAY2BGR)

    # Apply a colormap to the 3-channel image
    colorful_hog_image = cv2.applyColorMap(hog_image_3_channel, cv2.COLORMAP_JET)


    return colorful_hog_image


if __name__ == '__main__':
    # Initialize the paths
    img_folder = 'examples/easy'
    ann_path = 'examples/xray_test_easy.json'
    output_path = 'results'

    # Loading images and bboxes
    with open(ann_path, 'r') as file:
        coco_data = json.load(file)
    
    # Access the various components of the COCO format JSON
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']

    # Example 
    img_path = os.path.join(img_folder, images[2001]['file_name'])
    img = cv2.imread(img_path)
    bbox = annotations[2001]['bbox']
    enhanced_hog_img = hoggles(img, bbox)

    # Draw the bounding box on the image
    x, y, w, h = [int(coord) for coord in bbox]
    color = (0, 255, 0)  # Green color in BGR
    thickness = 4
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    cv2.rectangle(enhanced_hog_img, (x, y), (x + w, y + h), color, thickness)
    output_img = os.path.join(output_path, 'pred_4_img.png')
    output_hog = os.path.join(output_path, 'pred_4_hog.png')
    cv2.imwrite(output_img, img)
    cv2.imwrite(output_hog, enhanced_hog_img)

    #cv2.imshow('HOGgles Visualization', enhanced_hog_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()



# image_1
#bbox_img_1 = [215.4931506849315, 124.34246575342466, 77.39726027397262, 230.82191780821915]
#img_1_path = 'examples/easy/xray_easy00001.png'

#image = cv2.imread(img_1_path)
#enhanced_hog_image = hoggles(image, bbox_img_1)

#cv2.imshow('HOGgles Visualization', enhanced_hog_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()