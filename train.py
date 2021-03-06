import sys
import math
from sklearn import neighbors
import os
import os.path
import pickle
import argparse
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def knn_train(img_dir: str,
              model_path: str = None,
              n_neighbors: int = None,
              knn_algo: str = 'ball_tree',
              verbose: bool = False):
    """
    :param img_dir: Path to directory with images of faces. Directory structure should be like this:
        -img_dir
        --Person1
        ---img1.jpg
        ---img2.jpg
        ---...
        ---imgN.jpg
        --Person2
        ---img1.jpg
        ---img2.jpg
        ---...
        ---imgN.jpg
        --...
        --PersonN
        ---img1.jpg
        ---img2.jpg
        ---...
        ---imgN.jpg
    Supported images formats are 'png', 'jpg', 'jpeg'.
    :type img_dir: str

    :param model_path: Path and filename for storing trained KNN classifier.
    :type model_path: str

    :param n_neighbors: Number of neighbors for classifier train algorithm.
    :type n_neighbors: int

    :param knn_algo: Train algorithm's name.
    :type knn_algo: str

    :param verbose: Print additional information during execution?
    :type verbose: bool

    :return: Trained KNN classifier object.
    :rtype: KNeighborsClassifier
    """
    x = []
    y = []

    # For each class sub-directory in source directory
    for class_dir in os.listdir(img_dir):
        if not os.path.isdir(os.path.join(img_dir, class_dir)):
            continue

        # For each image in class sub-directory
        for image_path in image_files_in_folder(os.path.join(img_dir, class_dir)):
            image = face_recognition.load_image_file(image_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if not len(face_bounding_boxes) == 1:
                # Only one face should be in image, other cases will be skipped
                if verbose:
                    print("Cannot use {} for training: {}".format(image_path, "no face" if len(
                        face_bounding_boxes) < 1 else "too many faces"))
            else:
                # Add encoding of the found face to the training set
                x.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    if n_neighbors is None:
        # Set default neighbors count to use for weighting in the KNN classifier
        n_neighbors = int(round(math.sqrt(len(x))))
        if verbose:
            print("Default n_neighbors = {} used:".format(n_neighbors))

    # Train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(x, y)

    # Store the trained KNN classifier using pickle
    if model_path is not None:
        with open(model_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KNN classifier for face recognition."
                                                 "Using https://github.com/ageitgey/face_recognition")
    parser.add_argument('-ip', '--images-path', type=str, dest='img_dir', action='store',
                        required=True, help='path to images directory')
    parser.add_argument('-mp', '--model-path', type=str, dest='model_path', action='store',
                        default='trained_knn_model.clf',
                        help='file path to store classifier as pickle. Default is "trained_knn_model.clf"')
    args = parser.parse_args()

    print("Training KNN classifier...")
    classifier = knn_train(args.img_dir, model_path=args.model_path, n_neighbors=2)
    print("Training complete!")
    sys.exit(0)
