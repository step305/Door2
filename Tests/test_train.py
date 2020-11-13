import math
from sklearn import neighbors
import os
import os.path
import pickle
import argparse
import face_recognition
from PIL import Image, ImageDraw, ImageFont

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def predict(img_path, knn_clf=None, model_path=None, distance_threshold=0.4):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if not os.path.isfile(img_path) or os.path.splitext(img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either through knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    x_img = face_recognition.load_image_file(img_path)
    x_face_locations = face_recognition.face_locations(x_img)

    # If no faces are found in the image, return an empty result.
    if len(x_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(x_img, known_face_locations=x_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(x_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), x_face_locations, are_matches)]


def show_prediction_labels_on_image(img_path, predictions):
    """
    Shows the face recognition results visually.
    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """

    if predictions is None:
        return
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    w, h = pil_image.size
    font_size = int(14 * w/800)
    font_size = font_size if font_size > 18 else 18
    font = ImageFont.truetype("arial.ttf", font_size)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        if name == 'unknown':
            color_outline = (255, 0, 0)
        else:
            color_outline = (0, 255, 0)
        draw.rectangle(((left, top), (right, bottom)), outline=color_outline, width=3)

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name, font=font)
        draw.text((left + 6, bottom - int(1.2*text_height)), name, fill=(255, 255, 255, 255), font=font)

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    new_img_path = '.'.join(img_path.split('.')[0:-1]) + '_mod.jpeg'
    pil_image.save(new_img_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test KNN classifier for face recognition."
                                                 "Code is based on https://github.com/ageitgey/face_recognition")
    parser.add_argument('images', type=str, nargs='+', help='paths to images')
    parser.add_argument('--model-path', type=str, dest='model_path', action='store', required=True,
                        help='file path to store classifier as pickle')
    args = parser.parse_args()

    for img in args.images:
        preds = predict(img, model_path=args.model_path, distance_threshold=0.5)
        show_prediction_labels_on_image(img, preds)
