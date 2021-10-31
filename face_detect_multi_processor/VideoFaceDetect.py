import cv2
import time
import subprocess as sp
import multiprocessing as mp
from os import remove
import concurrent.futures

face_cascade = cv2.CascadeClassifier('dataset//haarcascade_frontalface_default.xml')


def multi_process(group_number, file_name, frame_jump_unit):
    # Read video file
    cap = cv2.VideoCapture(file_name)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_jump_unit * group_number)

    # get height, width and frame count of the video
    width, height = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    proc_frames = 0

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter()
    out.open("output_{}.mp4".format(group_number),fourcc, fps, (width, height), True)
    try:
        while proc_frames < frame_jump_unit:
            ret, frame = cap.read()
            if not ret:
                break

            im = frame
            # Perform face detection of frame
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Loop through list (if empty this will be skipped) and overlay green bboxes
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for (x, y, w, h) in faces:
                    cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # write the frame
            out.write(im)

            proc_frames += 1
    except Exception as e:
        # Release resources
        print(e)
        cap.release()
        out.release()
        exit(0)

    # Release resources
    cap.release()
    out.release()



def combine_output_files(num_processes):
    output_file_name = "video_Trim"+"_output_multi.mp4"
    
    # Create a list of output files and store the file names in a txt file
    list_of_output_files = ["output_{}.mp4".format(i) for i in range(num_processes)]
    with open("list_of_output_files.txt", "w") as f:
        for t in list_of_output_files:
            f.write("file {} \n".format(t))

    # use ffmpeg to combine the video output files
    ffmpeg_cmd = "ffmpeg -y -loglevel error -f concat -safe 0 -i list_of_output_files.txt -vcodec copy " + output_file_name
    sp.Popen(ffmpeg_cmd, shell=True).wait()

    # Remove the temperory output files
    for f in list_of_output_files:
        remove(f)
    remove("list_of_output_files.txt")


def VideoFaceDetect():

    file_name = "merc.mp4"
    width, height, frame_count = get_video_details(file_name)
    print("Video frame count = {}".format(frame_count))
    print("Width = {}, Height = {}".format(width, height))
    num_processes = 8
    print("Number of CPU: " + str(num_processes))
    frame_jump_unit = frame_count // num_processes

    print("Video processing using {} processes...".format(num_processes))
    start_time = time.time()
    # Paralle the execution of a function across multiple input values
    p = mp.Pool(num_processes)
    p.starmap(multi_process, [(i, file_name, frame_jump_unit) for i in range(num_processes)])
    combine_output_files(num_processes)
    p.close()
    end_time = time.time()

    total_processing_time = end_time - start_time
    print("Time taken: {}".format(total_processing_time))
    print("FPS : {}".format(frame_count/total_processing_time))
    

def get_video_details(file_name):
    cap = cv2.VideoCapture(file_name)
    length = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_HEIGHT)))
    return [width, height, length]

if __name__ == '__main__':
    VideoFaceDetect()
