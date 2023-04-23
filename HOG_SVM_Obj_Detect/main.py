import pickle
import argparse
from dataloader import *
from train import *
from hoggles import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HOG + SVM object detection.')
    parser.add_argument('-m', '--mode', required=True, help='type train to train the model, type predict to use the model to make prediction')
    #parser.add_argument('-i', '--image', required=True, help='Path to the input image.')
    #parser.add_argument('-m', '--model', required=True, help='Path to the trained HOG + SVM model in pkl format.')
    #parser.add_argument('-o', '--output', required=True, help='Path to save the output image with bounding boxes.')
    args = parser.parse_args()


    if args.mode == 'train':
        print("You are using the Training Mode ...")
        #==================== Training Process ====================
        # Training Dataset paths
        train_root = "/scratch/jp4906/pidray/train"
        train_annotation = "/scratch/jp4906/pidray/annotations/xray_train.json"

        # Load, parse, and extract HOG of the dataset
        print("Loading training dataset ... ")
        dataloader = PIDrayDataset(train_annotation, train_root, debug_mode=False)
        x_train_hog, y_train = dataloader.parse_data()
        print("Training dataset is loaded.")
        print("")

        # Train the SVM model
        print("Training the SVM model ... ")
        trained_model = train(x_train_hog, y_train)
        model_filename = '/scratch/jp4906/TSA-Prohibited-Obj-Detection/HOG_SVM_Obj_Detect/svm_model.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(trained_model, file)
        print("Training process is complete, and the trained model has been saved.")
        print("")


        #==================== Test Process ====================
        # Testing dataset paths
        test_root = "/scratch/jp4906/pidray/easy"
        test_annotation = "/scratch/jp4906/pidray/annotations/xray_test_easy.json"

        # Load, parse, and extract HOG of the dataset
        print("Loading test dataset ... ")
        dataloader = PIDrayDataset(test_annotation, test_root, debug_mode=False)
        x_test_hog, y_test = dataloader.parse_data()
        print("Test dataset is loaded.")
        print("")

        # Load the trained SVM model
        #with open(model_filename, 'rb') as file:
        #    loaded_clf = pickle.load(file)

        # Evaluate the trained SVM model
        print("Evaluating the trained model ... ")
        evaluation(x_test_hog, y_test, trained_model)
        print("Evalutation is completed, and a class-wise AP chart has been generated.")
        print("")

    elif args.mode == 'predict':
        print("You are using Prediction Mode ...")

        img_path = 'examples/pred_4.png'
        model_filename = 'trained-model/svm_model.pkl'

        print("Loading the pretrained model ...")
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)
        print("Model is loaded.")
        print("")
        