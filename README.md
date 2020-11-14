# Face recognition project on NVIDIA Jetson Nano
## Projects structure
Project based on **face_recognition** library: [face_recognition](https://github.com/ageitgey/face_recognition " ageitgey /
face_recognition ")
- ***train.py*** - script for  train KNN-classifier
- ***Tests/test_train.py*** - script for testing of trained KNN-classifier using images
- ***TestImgs/TrainImgs*** - sample faces for training of KNN-classifier 
- ***TestImgs/test...*** - images for testing of trained KNN-classifier using images
- ***TestImgs/trained_knn_model.clf*** - trained KNN-classifier

## Requirements
- [face_recognition library](https://github.com/ageitgey/face_recognition)
- Python (tested 3.8 on Windows 10)
- [Google Coral Runtime](https://coral.ai/docs/accelerator/get-started)
- [Google PyCoral library](https://github.com/google-coral/pycoral/)
- Google Coral USB Accelerator
- [DLIB](http://dlib.net/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [Pillow library](https://github.com/python-pillow/Pillow) 

### Note
Model for Google Coral can be downloaded from official site:
[Edge TPU MobileNet SSD v2 (Faces)](https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite))
It should be save to model sub-directory. Name it **ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite**.


### ToDo:
- face detection function using Google Coral USB
- face recognition function from webcam
- porting code to Jetson Nano
 
