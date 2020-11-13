#F ace recognition project on NVIDIA Jetson Nano
## Projects structure
Project based on **face_recognition** library: [https://github.com/ageitgey/face_recognition](https://github.com/ageitgey/face_recognition " ageitgey /
face_recognition ")
- ***train.py*** - script for  train KNN-classifier
- ***Tests/test_train.py*** - script for testing of trained KNN-classifier using images
- ***TestImgs/TrainImgs*** - sample faces for training of KNN-classifier 
- ***TestImgs/test...*** - images for testing of trained KNN-classifier using images
- ***TestImgs/trained_knn_model.clf*** - trained KNN-classifier

### ToDo:
- face detection function using Google Coral USB
- face recognition function from webcam
- porting code to Jetson Nano
 