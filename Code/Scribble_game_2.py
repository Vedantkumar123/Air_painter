import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np
button_regions = [
    {'start': (0, 0), 'end': (100, 50), 'color': (0, 0, 255), 'text': 'Red','text_color': (0, 0, 0)},
    {'start': (110, 0), 'end': (210, 50), 'color': (0, 255, 0), 'text': 'Green','text_color': (0, 0, 0)},
    {'start': (220, 0), 'end': (320, 50), 'color': (255, 0, 0), 'text': 'Blue','text_color': (0, 0, 0)},
    {'start': (330, 0), 'end': (430, 50), 'color': (255, 255, 255), 'text': 'Eraser','text_color': (0, 0, 0)}
]

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
line_number=0
annotations_start = False
line_color=(0,255,0)
while True:
    success, img = cam.read()
    img = cv2.flip(img, 1)
    path_full_image = os.path.join(folderpath, pathimages[img_number])
    img_current = cv2.imread(path_full_image)
    img_current = cv2.resize(img_current, (1080, 720))
    hands, img = detector.findHands(img)
    # cv2.line(img, (0, gesture_threshold), (width, gesture_threshold), (0, 255, 0), 10)
    for button in button_regions:
        cv2.rectangle(img_current, button['start'], button['end'], button['color'], -1)
        cv2.rectangle(img_current, button['start'], button['end'], (0, 0, 0), 2)
        cv2.putText(img_current, button['text'], (button['start'][0] + 10, button['start'][1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, button['text_color'], 2, cv2.LINE_AA)

    if hands and button_pressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmList = hand['lmList']

        xval = int(np.interp(lmList[8][0], [width // 1.8, width - 150], [0, width]))
        yval = int(np.interp(lmList[8][1], [150, height - 300], [0, height]))
        indexfinger = xval, yval
        # print(fingers)
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
        if fingers == [0, 0, 0, 0, 1]:
            print("clear")
            annotations = [[]]
            annotations_number = 0
            annotations_start = False
        if fingers == [0, 1, 0, 0, 0]:
            print("pointing")
            cv2.circle(img_current, indexfinger, 8, (0, 0, 255), cv2.FILLED)
            for i, button in enumerate(button_regions):
                start, end = button['start'], button['end']
                if start[0] < xval < end[0] and start[1] < yval < end[1]:
                    # If index finger is over the button, change the line color
                    # line_number += 1
                    line_color = button['color']
                    # button_hovered = i
                # annotations[annotations_number].append([])

        if fingers == [0, 1, 1, 0, 0]:
            print("drawing")
            if annotations_start == False:
                annotations_start = True
                annotations_number += 1
                annotations.append([])
                line_number = 0
            # cv2.circle(img_current, indexfinger, 8, (0, 255, 0), cv2.FILLED)
            cv2.circle(img_current, indexfinger, 8, line_color, cv2.FILLED)
            annotations[annotations_number].append([])
            annotations[annotations_number][line_number].append(indexfinger)
            annotations[annotations_number][line_number].append(line_color)
            line_number += 1
        else:
            annotations_start = False
        if fingers == [0, 1, 1, 1, 0]:
            if annotations_number > 0:
                redoannotations.append(annotations[-1])
                # print(redoannotations)
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
        for j in range(1, len(annotations[i])):
            if j != 0 or len(annotations[i][j]) != 0 or len(annotations[i][j-1])!=0 or len(annotations[i][j+1]!=0):
                cv2.line(img_current, annotations[i][j-1][0], annotations[i][j][0], annotations[i][j-1][1], 12)
                # cv2.line(img_current, annotations[i][j - 1], annotations[i][j], annotations[i][j], 12)
                # print(annotations)
                # print(annotations[i][j])
    img_small = cv2.resize(img, (ws, hs))
    h, w, _ = img_current.shape
    img_current[0:hs, w - ws:w] = img_small
    # cv2.imshow("image",img)
    cv2.imshow("slide", img_current)
    key = cv2.waitKey(1)
    if key == 27:
        break
