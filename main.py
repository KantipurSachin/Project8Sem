#Starter codes 
import cv2
import dlib

print("open CV version: "+cv2.__version__)
print("Dlib version: "+dlib.__version__)

# Load the face detector and landmark predictor models
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('D:/project/DlibPackages/dlib_for_python_3.9/shape_predictor_68_face_landmarks.dat')

# Open the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_detector(gray)

    # Iterate over detected faces
    for face in faces:
        # Get the facial landmarks
        landmarks = landmark_predictor(gray, face)

        # Draw a rectangle around the face
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Iterate over the facial landmarks
        for i in range(68):
            # Get the coordinates of the facial landmark points
            x = landmarks.part(i).x
            y = landmarks.part(i).y

            # Draw a circle for each facial landmark
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    # Display the frame with detections
    cv2.imshow('Lakpa"s Drowsiness Detection App', frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video_capture.release()
cv2.destroyAllWindows()
