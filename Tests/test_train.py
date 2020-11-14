import sys
from sklearn import neighbors
import os
import os.path
import pickle
import argparse
import face_recognition
from PIL import Image, ImageDraw, ImageFont

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def classify_faces(img_path: str,
                   knn_clf: neighbors.KNeighborsClassifier = None,
                   model_path: str = None,
                   distance_threshold: float = 0.4):
    """
    Try to classify faces in given image using a trained KNN classifier.
    :param img_path: path to image to be processed.
    :type img_path: str

    :param knn_clf: (optional) a KNN classifier object. If not specified, model_save_path must be specified.
    :type knn_clf: neighbors.KNeighborsClassifier

    :param model_path: (optional) path to a pickled KNN classifier. If not specified, knn_clf must be specified.
    :type model_path: str

    :param distance_threshold: (optional) distance threshold for face classification. Value 0...1.
    Less value requires more strict math of faces during classification.
    :type distance_threshold: float

    :return: a list of names and faces locations for the recognized faces in the source image:
    [(name, bounding box), ...]. For faces of unrecognized persons, the name 'unknown' will be returned.
    :rtype: [(str, (int, int, int, int))]
    """
    assert 0.0 <= distance_threshold <= 1.0, "distance_threshold should be in range 0.0 .. 1.0"
    if not os.path.isfile(img_path) or os.path.splitext(img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(img_path))

    if knn_clf is None and model_path is None:
        raise Exception("KNN classifier not found neither through knn_clf, neither through model_path")

    # Try load trained KNN classifier
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image and detect face locations
    x_img = face_recognition.load_image_file(img_path)
    x_face_locations = face_recognition.face_locations(x_img)

    # If no faces there are in the image, return an empty result.
    if len(x_face_locations) == 0:
        return []

    # Calculate encodings for faces in the source image
    faces_encodings = face_recognition.face_encodings(x_img, known_face_locations=x_face_locations)

    # Find the best match of found faces using KNN classifier
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(x_face_locations))]

    # Return only names that have closeness within the threshold, otherwise return name="unknown"
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), x_face_locations, are_matches)]


def draw_faces_boxes(image_path: str,
                     faces: [(str, (int, int, int, int))]):
    """
    Draw faces surrounding boxes on image. Sign faces. Save modified image in the new JPEG file (adding "_mod" to name).
    :param image_path: path to image to be processed
    :type image_path: str

    :param faces: results of the classify_faces function
    :type faces: [(str, (int, int, int, int))]

    :return:
    """
    if faces is None:
        return
    image = Image.open(image_path).convert("RGB")
    layout = ImageDraw.Draw(image)
    width, _ = image.size
    font_size = int(14 * width / 800)
    font_size = font_size if font_size > 18 else 18
    font = ImageFont.truetype("arial.ttf", font_size)

    for name, (top, right, bottom, left) in faces:
        # Draw a box around the face
        if name == 'unknown':
            # Unknown faces should be surrounded with red box
            color_outline = (255, 0, 0)
        else:
            # Known faces should be surrounded with green box
            color_outline = (0, 255, 0)
        layout.rectangle(((left, top), (right, bottom)), outline=color_outline, width=3)

        # Sign the face
        text_width, text_height = layout.textsize(name, font=font)
        layout.text((left + 0.1*text_width, bottom - int(1.2 * text_height)),
                    name, fill=(255, 255, 255, 255), font=font)

    # Remove the drawing library from memory as recommended by the Pillow docs
    del layout

    # Save modified image.
    new_img_path = '.'.join(image_path.split('.')[0:-1]) + '_mod.jpeg'
    image.save(new_img_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test KNN classifier for face recognition."
                                                 "Code is based on https://github.com/ageitgey/face_recognition")
    parser.add_argument('images', type=str, nargs='+', help='paths to images')
    parser.add_argument('-mp', '--model-path', type=str, dest='model_path', action='store', required=True,
                        help='file path to store classifier as pickle')
    args = parser.parse_args()

    for img in args.images:
        found_faces = classify_faces(img, model_path=args.model_path, distance_threshold=0.5)
        draw_faces_boxes(img, found_faces)

    sys.exit(0)
