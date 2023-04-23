import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import accuracy_score, make_scorer


def train(x_train_hog, y_train):
    params_grid = [
        {'C': [1, 10, 100], 'kernel': ['linear']},
        {'C': [1, 10, 100], 'gamma': [0.1, 0.01], 'kernel': ['rbf']}
    ]

    svc = SVC()
    clf = GridSearchCV(svc, params_grid, cv=5)
    clf.fit(x_train_hog, y_train)
    return clf


def compute_classwise_ap_and_map(y_true, y_pred, num_classes=12):
    aps = []

    for class_id in range(num_classes):
        # Binary classification of the current class
        y_true_binary = (y_true == class_id).astype(int)
        y_pred_binary = (y_pred == class_id).astype(int)

        # Calculate precision-recall curve and AUC (Average Precision)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_binary)
        ap = auc(recall, precision)
        aps.append(ap)

    map_score = np.mean(aps)

    return aps, map_score


def plot_classwise_ap(classwise_ap):
    num_classes = len(classwise_ap)
    class_labels = [f'Class {i + 1}' for i in range(num_classes)]
    class_names = ['Baton', 'Pliers', 'Hammer', 'Powerbank', 'Scissors', 'Wrench', 'Gun', 'Bullet', 'Sprayer', 'HandCuffs', 'Knife', 'Lighter']

    fig, ax = plt.subplots()
    ax.bar(class_labels, classwise_ap)

    ax.set_xlabel('Classes')
    ax.set_ylabel('Average Precision')
    ax.set_title('Class-wise Average Precision')
    plt.xticks(np.arange(num_classes), class_names, rotation='vertical')

    # Save the chart as a PDF
    output_file="/scratch/jp4906/TSA-Prohibited-Obj-Detection/HOG_SVM_Obj_Detect/ap_chart.pdf"
    plt.savefig(output_file, format='pdf')


def evaluation(x_test_hog, y_test, clf):
    # Generate the prediction
    y_pred = clf.predict(x_test_hog)

    # Claculate class-wise AP and mAP
    classwise_ap, map_score = compute_classwise_ap_and_map(y_test, y_pred)
    print("mAP Score: {}".format(map_score))

    # Plot the class-wise AP
    plot_classwise_ap(classwise_ap)