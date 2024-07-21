import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np

detector = HandDetector(detectionCon=0.8, maxHands=1)
width = 1080
height = 720
folderpath = r"C:\Users\KIIT\OneDrive\Desktop\Vedant_Official\vedant_projects\Deep learning\hand_gesture\presentation"
cam = cv2.VideoCapture(0)
# cam=cv2.VideoCapture("http://192.168.165.92:8080/video")
cam.set(3, width)
cam.set(4, height)
pathimages = sorted(os.listdir(folderpath), key=len)
# print(pathimages)

img_number = 0
hs, ws = int(120 * 1.5), int(213 * 1.5)
gesture_threshold = 300
button_pressed = False
button_counter = 0
button_delay = 10
annotations = [[]]
redoannotations = [[]]
annotations_number = 0
threshold = 0  # Adjust as needed
annotations_start = False

while True:
    success, img = cam.read()
    img = cv2.flip(img, 1)
    path_full_image = os.path.join(folderpath, pathimages[img_number])
    img_current = cv2.imread(path_full_image)
    img_current = cv2.resize(img_current, (1080, 720))
    hands, img = detector.findHands(img)
    cv2.line(img, (0, gesture_threshold), (width, gesture_threshold), (0, 255, 0), 10)
    if hands and button_pressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmList = hand['lmList']

        xval = int(np.interp(lmList[8][0], [width // 1.8, width - 150], [0, width]))
        yval = int(np.interp(lmList[8][1], [150, height - 300], [0, height]))
        indexfinger = xval, yval
        print(fingers)
        # if cy<=gesture_threshold:
        #     # pass
        #     if fingers == [0, 1, 0, 0, 0]:
        #         print("pointing")
        #         button_pressed=True
        #         if img_number>0:
        #             img_number-=1
        #     if fingers == [0, 1, 1, 0, 0]:
        #         print("drawing")
        #         button_pressed=True
        #         if img_number<len(pathimages)-1:
        #             img_number+=1
        if fingers == [0, 0, 0, 0, 1]:
            print("clear")
            annotations = [[]]
            annotations_number = 0
            annotations_start = False
        if fingers == [0, 1, 0, 0, 0]:
            print("pointing")
            cv2.circle(img_current, indexfinger, 8, (0, 0, 255), cv2.FILLED)

        if fingers == [0, 1, 1, 0, 0]:
            print("drawing")
            if annotations_start is False:
                annotations_start = True
                annotations_number += 1
                annotations.append([])
            cv2.circle(img_current, indexfinger, 8, (0, 255, 0), cv2.FILLED)
            annotations[annotations_number].append(indexfinger)
            if annotations[annotations_number]:
                if np.linalg.norm(np.array(indexfinger) - np.array(annotations[annotations_number][-1])) > threshold:
                    annotations[annotations_number].append(indexfinger)
            else:
                annotations[annotations_number].append(indexfinger)

            # Add a new position to the annotations list

            # Smooth the position by averaging the last few positions
            # if annotations[annotations_number]:
            #     smoothed_position = np.mean(annotations[annotations_number][-5:], axis=0).astype(int)
            #     annotations[annotations_number].pop(-1)
            #     annotations[annotations_number].append(smoothed_position)
            #     cv2.circle(img_current, tuple(smoothed_position), 8, (0, 255, 0), cv2.FILLED)
            #
            #     # Draw lines using smoothed positions
            #     smoothed_positions = annotations[annotations_number]
            #     for j in range(1, len(smoothed_positions)):
            #         cv2.line(img_current, tuple(smoothed_positions[j - 1]), tuple(smoothed_positions[j]),
            #                  (204, 204, 51), 12)
            # else:
            #     annotations[annotations_number].append(indexfinger)

        else:
            annotations_start = False

        if fingers == [0, 1, 1, 1, 0]:
            if annotations_number > 0:
                redoannotations.append(annotations[-1])
                annotations.pop(-1)
                annotations_number -= 1
                button_pressed = True
        if fingers == [0, 1, 1, 1, 1]:
            if len(redoannotations) > 0:
                annotations.append(redoannotations[-1])
                redoannotations.pop(-1)
                annotations_number += 1
                button_pressed = True
    else:
        annotations_start = False
    if button_pressed:
        button_counter += 1
        if button_counter > button_delay:
            button_counter = 0
            button_pressed = False
    for i in range(0, len(annotations)):
        for j in range(0, len(annotations[i])):
            if j != 0:
                cv2.line(img_current, annotations[i][j - 1], annotations[i][j], (204, 204, 51), 12)
    img_small = cv2.resize(img, (ws, hs))
    h, w, _ = img_current.shape
    img_current[0:hs, w - ws:w] = img_small
    # cv2.imshow("image",img)
    cv2.imshow("slide", img_current)
    key = cv2.waitKey(1)
    if key == 27 or key == ord("q"):
        break
