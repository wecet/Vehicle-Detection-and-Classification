import cv2
import numpy as np
from imutils.video import VideoStream

def detect_background(file):
    """Extracts background and saves it to a jpg and avi video file"""

    vid = cv2.VideoCapture(file) 
    #subtractor = cv2.createBackgroundSubtractorMOG2() #Dan isu harira ahjar
    subtractor = cv2.createBackgroundSubtractorKNN()

    bg_frames = []
    i = 0

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter('background.avi', fourcc, 30, (800, 600), True)

    ret = True
    while(ret):
        ret, frame = vid.read()

        mask = subtractor.apply(frame)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask = cv2.bitwise_not(mask) #To change background to white


        combined = cv2.bitwise_and(frame, mask)
        bg_frames.append(combined)

        i+=1
        if i == 300:
            break

    medianFrame = np.median(bg_frames, axis=0).astype(dtype=np.uint8)
    cv2.imwrite("extracted_background.jpg", medianFrame)
    cv2.imshow("Median Frame", medianFrame)
    cv2.waitKey(0)

    bg_frames = [cv2.addWeighted(frame, 0.1, medianFrame, 0.9, 0) for frame in bg_frames]

    for frame in bg_frames:
        writer.write(cv2.resize(frame,(800, 600)))


    writer.release()
    vid.release()
    cv2.destroyAllWindows()

detect_background("20200323_155250.mp4")