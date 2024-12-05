# import streamlit as st
# import cv2
# import numpy as np
# from keras.models import load_model

# # Load your pre-trained model
# model_path = "my_model_64p35.h5"
# new_model = load_model(model_path)

# # Define labels for emotion classes
# EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# # Function for predicting emotion
# def predict_emotion(face_roi):
#     final_image = cv2.resize(face_roi, (224, 224))
#     final_image = np.expand_dims(final_image, axis=0)
#     final_image = final_image / 255.0
#     predictions = new_model.predict(final_image)
#     return EMOTIONS[np.argmax(predictions)]

# # Streamlit App Interface
# st.title("Face Emotion Recognition")

# # File upload for image input
# uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Load and display the image
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     frame = cv2.imdecode(file_bytes, 1)
#     st.image(frame, channels="BGR", caption="Uploaded Image")

#     # Convert to grayscale and detect faces
#     faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

#     if len(faces) > 0:
#         for (x, y, w, h) in faces:
#             roi_gray = gray[y:y+h, x:x+w]
#             roi_color = frame[y:y+h, x:x+w]
#             emotion = predict_emotion(roi_color)

#             # Draw rectangle and label
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#             cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#         st.image(frame, channels="BGR", caption="Processed Image")
#     else:
#         st.warning("No faces detected in the image.")








# import streamlit as st
# import cv2
# import numpy as np
# from keras.models import load_model

# # Load your pre-trained model
# model_path = "my_model_64p35.h5"
# new_model = load_model(model_path)

# # Define emotion labels
# emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# # Load Haar Cascade for face detection
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Streamlit App Interface
# st.title("Real-Time Face Emotion Recognition")
# st.write("Use your webcam to detect emotions!")

# # Webcam input through Streamlit
# run = st.button("Start Webcam")

# if run:
#     # Start capturing video from the webcam
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         st.error("Cannot open webcam")
#     else:
#         while True:
#             ret, frame = cap.read()

#             if not ret:
#                 st.error("Failed to grab frame")
#                 break

#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

#             for x, y, w, h in faces:
#                 roi_color = frame[y:y+h, x:x+w]

#                 # Preprocess the face ROI for the model
#                 try:
#                     final_image = cv2.resize(roi_color, (224, 224))  # Model input size
#                     final_image = np.expand_dims(final_image, axis=0)
#                     final_image = final_image / 255.0

#                     predictions = new_model.predict(final_image)

#                     # Emotion classification
#                     predicted_class = np.argmax(predictions)
#                     status = emotions[predicted_class]
#                     confidence = predictions[0][predicted_class]

#                     # Draw rectangle and emotion label
#                     x1, y1, w1, h1 = x, y, 175, 75  # Coordinates for status box
#                     cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
#                     cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

#                 except Exception as e:
#                     st.error(f"Error processing frame: {e}")

#             # Display the resulting frame
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Streamlit
#             st.image(frame_rgb, channels="RGB", use_column_width=True)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()










# import streamlit as st
# import cv2
# import numpy as np

# # Load your pre-trained model (ensure it's placed correctly in your project directory)
# from keras.models import load_model
# model_path = 'my_model_64p35.h5'
# new_model = load_model(model_path)

# # Define emotion labels
# EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# # Function for predicting emotion
# def predict_emotion(face_roi):
#     final_image = cv2.resize(face_roi, (224, 224))
#     final_image = np.expand_dims(final_image, axis=0)
#     final_image = final_image / 255.0
#     predictions = new_model.predict(final_image)
#     return EMOTIONS[np.argmax(predictions)]

# st.title("Face Emotion Recognition")

# # Camera input
# frame = st.camera_input("Capture your face")

# if frame is not None:
#     # Convert frame to OpenCV format
#     file_bytes = np.asarray(bytearray(frame.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, 1)

#     # Convert to grayscale and detect faces
#     faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(gray, 1.1, 4)

#     for (x, y, w, h) in faces:
#         roi_color = img[y:y+h, x:x+w]
#         emotion = predict_emotion(roi_color)

#         # Draw rectangle and label
#         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # Display the processed image
#     st.image(img, channels="BGR", caption="Processed Image")

















# import streamlit as st
# import cv2
# import numpy as np
# from keras.models import load_model

# # Load your pre-trained emotion recognition model
# model_path = "my_model_64p35.h5"  # Ensure this is the correct path to your model
# new_model = load_model(model_path)

# # Define labels for emotion classes
# EMOTIONS = ["angry","disgust","fear","happy","neutral","sad","surprise"]

# # Function to predict emotion
# def predict_emotion(face_roi):
#     final_image = cv2.resize(face_roi, (224, 224))  # Resize to (224, 224)
#     final_image = np.expand_dims(final_image, axis=0)
#     final_image = final_image / 255.0
#     predictions = new_model.predict(final_image)
#     return EMOTIONS[np.argmax(predictions)]

# # Streamlit App Interface
# # st.title("Real-time Face Emotion Recognition")
# st.markdown("<h1 style='text-align: center;'><u>Real-time Face Emotion Recognition</u></h1>", unsafe_allow_html=True)

# # Create a placeholder for the video stream
# frame_window = st.image([])  # Placeholder to display frames

# # Start the video stream
# cap = cv2.VideoCapture(0)

# # Ensure the webcam is opened
# if not cap.isOpened():
#     cap = cv2.VideoCapture(0)

# # Load OpenCV face cascade for face detection
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         st.error("Failed to grab frame.")
#         break

#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the image
#     faces = faceCascade.detectMultiScale(gray, 1.1, 4)

#     # Process each detected face
#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]

#         # Predict emotion for the detected face
#         emotion = predict_emotion(roi_color)

#         # Draw bounding box and emotion label
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # Display the resulting frame on Streamlit
#     frame_window.image(frame, channels="BGR", use_container_width=True)

#     # Break the loop if 'q' is pressed (but Streamlit app should run continuously)
#     # To stop the video stream properly, you can add logic to quit using the "stop" button in Streamlit

# cap.release()
# cv2.destroyAllWindows()


















import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Load your pre-trained emotion recognition model
model_path = "my_model_64p35.h5"  # Ensure this is the correct path to your model
new_model = load_model(model_path)

# Define labels for emotion classes
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Function to predict emotion
def predict_emotion(face_roi):
    final_image = cv2.resize(face_roi, (224, 224))  # Resize to (224, 224)
    final_image = np.expand_dims(final_image, axis=0)
    final_image = final_image / 255.0
    predictions = new_model.predict(final_image)
    return EMOTIONS[np.argmax(predictions)]

# Streamlit App Interface
st.markdown("<h1 style='text-align: center;'><u>Real-time Face Emotion Recognition</u></h1>", unsafe_allow_html=True)

# Create a placeholder for the video stream
frame_window = st.image([])  # Placeholder to display frames

# Start the video stream
cap = cv2.VideoCapture(0)

# Ensure the webcam is opened
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

# Load OpenCV face cascade for face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def draw_rounded_rectangle(image, top_left, bottom_right, color, thickness, radius):
    """
    Draw a rectangle with rounded corners on an image.
    """
    # Top-left corner
    cv2.ellipse(image, (top_left[0] + radius, top_left[1] + radius), (radius, radius), 180, 0, 90, color, thickness)
    # Top-right corner
    cv2.ellipse(image, (bottom_right[0] - radius, top_left[1] + radius), (radius, radius), 270, 0, 90, color, thickness)
    # Bottom-left corner
    cv2.ellipse(image, (top_left[0] + radius, bottom_right[1] - radius), (radius, radius), 90, 0, 90, color, thickness)
    # Bottom-right corner
    cv2.ellipse(image, (bottom_right[0] - radius, bottom_right[1] - radius), (radius, radius), 0, 0, 90, color, thickness)

    # Draw straight lines to complete the rectangle
    cv2.line(image, (top_left[0] + radius, top_left[1]), (bottom_right[0] - radius, top_left[1]), color, thickness)
    cv2.line(image, (top_left[0] + radius, bottom_right[1]), (bottom_right[0] - radius, bottom_right[1]), color, thickness)
    cv2.line(image, (top_left[0], top_left[1] + radius), (top_left[0], bottom_right[1] - radius), color, thickness)
    cv2.line(image, (bottom_right[0], top_left[1] + radius), (bottom_right[0], bottom_right[1] - radius), color, thickness)

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to grab frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    # Process each detected face
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Predict emotion for the detected face
        emotion = predict_emotion(roi_color)

        # Draw rounded rectangle with emotion label
        draw_rounded_rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3, 15)  # Adjust thickness and radius as needed
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame on Streamlit
    frame_window.image(frame, channels="BGR", use_container_width=True)

cap.release()
cv2.destroyAllWindows()
