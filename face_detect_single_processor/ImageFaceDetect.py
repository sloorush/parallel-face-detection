import cv2
import time

def ImageFaceDetect(fn,filename):
    start_time = time.time()
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('dataset//haarcascade_frontalface_default.xml')
    # Read the input image
    img = cv2.imread(filename)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    end_time = time.time()
    print(fn," single time = ",(end_time-start_time))
    print("No of faces = ", len(faces))
    # Display the output
    cv2.imshow('img', img)
    cv2.waitKey()
    del face_cascade
    del img
    del gray
    del faces

if __name__ == '__main__':
    fn = "crowdFace"
    file_name = fn + ".jpg"
    ImageFaceDetect(fn, file_name)
