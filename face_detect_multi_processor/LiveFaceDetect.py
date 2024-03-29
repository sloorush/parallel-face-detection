import cv2
import time
import concurrent.futures


def LiveFaceDetect():
    # Load the cascade
    face_cascade = cv2.CascadeClassifier("dataset//haarcascade_frontalface_default.xml")

    # To capture video from webcam.
    cap = cv2.VideoCapture(0)

    frm = 0
    start_time = time.time()
    while True:
        frm = frm + 1
        # Read the frame
        _, img = cap.read()
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw the rectangle around each face
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Display
        if frm == 10:
            end_time = time.time()
            dur_time = end_time - start_time
            print("fps = ", (10 / dur_time))
            frm = 0
            start_time = time.time()
        cv2.imshow("img", img)
        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
    # Release the VideoCapture object
    cap.release()


LiveFaceDetect()
