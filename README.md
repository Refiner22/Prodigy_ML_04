# Prodigy_ML_04
## **Gesture Recognition using Deep Learning**

# ***Overview***
* This project is designed to recognize hand gestures using computer vision and deep learning.
* The system leverages **OpenCV** for image processing, **Mediapipe** for hand tracking, and **TensorFlow** for model training and inference.
* It captures hand gestures in real-time, preprocesses the data, and trains a Convolutional Neural Network (CNN) to classify different gestures.

## ***Features***
* **Real-time Gesture Detection** - Utilizes OpenCV and Mediapipe for real-time hand tracking.
* **Dataset Collection & Preprocessing** - Collects gesture images and converts them into a structured dataset.
* **Deep Learning Model** - Implements a CNN to classify gestures accurately.
* **User Interaction** - Allows users to provide hand gestures for training and recognition.
* **Scalability** - The model can be expanded to support additional gesture classes.

# ***Installation***
## * **Step 1:** Clone the repository
  ```bash
  git clone <repository_url>
  cd <repository_folder>
  ```
## * **Step 2:** Install the required dependencies
  ```bash
  pip install -r requirements.txt
  ```
## * **Step 3:** Run the dataset collection script to capture gestures
  ```bash
  python collect_gestures.py
  ```
## * **Step 4:** Train the model with the collected data
  ```bash
  python train_model.py
  ```
## **Step 5:** Run the real-time gesture recognition script
  ```bash
  python recognize_gestures.py
  ```

# ***Usage Guide***
## * **Collect Gestures:**
  - Run the `collect_gestures.py` script to capture hand gesture images.
  - Press **'c'** to start capturing images.
  - Press **'q'** to stop and save the dataset.

## **Train the Model:**
  - Execute `train_model.py` to train a CNN on the collected gesture images.
  - The trained model is saved as `model.h5`.

## **Run Gesture Recognition:**
  - Use `recognize_gestures.py` to classify gestures in real-time.
  - The model predicts the gesture and displays the result.

# ***Project Structure***
* **`collect_gestures.py`** - Captures and saves gesture images.
* **`train_model.py`** - Trains a CNN model for classification.
* **`recognize_gestures.py`** - Uses the trained model to recognize gestures in real-time.
* **`model.h5`** - Saved deep learning model file.

# ***Dependencies***
* **Python 3.7+**
* **OpenCV** - For image capturing and processing.
* **Mediapipe** - For hand landmark detection.
* **TensorFlow & Keras** - For building and training the CNN model.
* **NumPy** - For handling numerical operations.

# ***Future Enhancements***
* Add support for additional gestures.
* Improve model accuracy with more training data.
* Implement a mobile application for gesture recognition.



