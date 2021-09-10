import cv2 as cv
import mediapipe as mp
import time

# To open the camera for video capturing.
cap = cv.VideoCapture(0)


mpHands = mp.solutions.hands

hands = mpHands.Hands()

# To draw lines on the hand
mpDraw = mp.solutions.drawing_utils

# Initializing variables to calculate frame rate
old_Time = 0
current_Time = 0


while True:
    success, img = cap.read()
    # Converting image into RGB as per class requirement
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # Using process method from the Hands class
    results = hands.process(imgRGB)
    # Detect the number of hands
    #print(results.multi_hand_landmarks)

    # if the model detects a hand then
    if results.multi_hand_landmarks:
        # then for each hand detected
        for handLms in results.multi_hand_landmarks:
            # id and lm denote index and xyz pixel coordinates (for finger volume detection)
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                # height width, channels
                h, w, c = img.shape
                # The coordinates are in decimals hence require conversion to their pixel form
                cx , cy = int(lm.x * w), int(lm.y*h)
                # id, cx and cy
                print(id, cx, cy)
                # Id 4 represents the tip of the thumb
                #if id == 4:
                cv.circle(img, (cx,cy), 10, (255,255, 0), cv.FILLED)


            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    current_Time = time.time()
    fps = 1 / (current_Time - old_Time)
    old_Time = current_Time

    cv.putText(img, str(int(fps)), (10, 70),cv.FONT_HERSHEY_PLAIN, 3,
    (255, 255, 255), 3 )

    cv.imshow("Image", img)
    cv.waitKey(1)



