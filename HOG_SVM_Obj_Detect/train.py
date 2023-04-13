import numpy as np
from sklearn.svm import SVC
from skimage import feature
from sklearn.metrics import average_precision_score
from dataloader import dataloader


def hog_feature_extractor(img):
    """This function computes the HOG feature vector for a given image

    Args:
        img (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Convert image to grayscale
    gray = np.array(img.convert('L'))
    # Compute HOG features
    hog_feats = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
    return hog_feats


def train(root, annotation, epochs=10):
    train_loader, val_loader = dataloader(root, annotation, train=True)

    # Loop over multiple epochs
    for epoch in range(epochs):
        # Loop over train dataloader
        for batch_idx, (imgs, targets) in enumerate(train_loader):
            # Extract HOG features and labels for batch
            batch_feats = []
            batch_labels = []
            for i in range(len(imgs)):
                hog_feats = hog_feature_extractor(imgs[i])
                batch_feats.append(hog_feats)
                batch_labels.extend(targets[i]['labels'].tolist())

            # Train SVM classifier on batch
            clf = SVC(kernel='linear')
            clf.fit(batch_feats, batch_labels)

            # Evaluate on validation set after every 10 batches
            if batch_idx % 10 == 0:
                val_feats = []
                val_labels = []
                for img, target in val_loader:
                    hog_feats = hog_feature_extractor(img)
                    val_feats.append(hog_feats)
                    val_labels.extend(target['labels'].tolist())

                val_preds = clf.predict(val_feats)
                ap = average_precision_score(val_labels, val_preds)
                print('Epoch: {}, Batch: {}, Average Precision: {:.4f}'.format(epoch, batch_idx, ap))