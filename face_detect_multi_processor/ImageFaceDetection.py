import cv2
import time
import concurrent.futures

def ImageFaceDetect(fn, file_name):
    start_time = time.time()
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('dataset//haarcascade_frontalface_default.xml')
    # Read the input image
    img = cv2.imread(file_name)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    end_time = time.time()
    exec_time = end_time  - start_time
    print(fn, "_multiprocessing time = ", exec_time)
    print("No of faces = ",len(faces))
    
    # Display the output
    # cv2.imshow('img', img)
    # cv2.waitKey()

if __name__ == '__main__':
    fn = "crowdFace"
    file_name = fn + ".jpg"
    ImageFaceDetect(fn,file_name)
    
