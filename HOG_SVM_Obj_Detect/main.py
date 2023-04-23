import pickle
from dataloader import *
from train import *


if __name__ == "__main__":

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