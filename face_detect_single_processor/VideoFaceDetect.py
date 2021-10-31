import cv2
import time
import subprocess as sp
import multiprocessing as mp
from os import remove
face_cascade = cv2.CascadeClassifier('dataset//haarcascade_frontalface_default.xml')


def process_video():
    # Read video file
    cap = cv2.VideoCapture(file_name)

    # get height, width and frame count of the video
    width, height = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter()
    out.open(output_file_name, fourcc, fps, (width, height), True)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            im = frame
            # Perform face detection of frame
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray,1.3,5)

            # Loop through list (if empty this will be skipped) and overlay green bboxes
            
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # write the frame
            out.write(im)
    except Exception as e:
        # Release resources
        print(e)
        cap.release()
        out.release()

    # Release resources
    cap.release()
    out.release()


def VideoFaceDetect() :
    print("Video processing using single process...")
    start_time = time.time()
    process_video()
    end_time = time.time()
    total_processing_time = end_time - start_time
    print("Time taken: {}".format(total_processing_time))
    print("Video frame count = {}".format(frame_count))
    print("FPS : {}".format(frame_count/total_processing_time))


def get_video_details(file_name):
    cap = cv2.VideoCapture(file_name)
    length = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_HEIGHT)))
    return [width, height, length]


fn = "merc"
file_name = fn+".mp4"
output_file_name = fn+"_output_single.mp4"
width, height, frame_count = get_video_details(file_name)
print("Video frame count = {}".format(frame_count))
print("Width = {}, Height = {}".format(width, height))
VideoFaceDetect()
