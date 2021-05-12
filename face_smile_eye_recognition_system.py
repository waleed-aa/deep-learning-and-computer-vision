#Importing opencv module
import cv2 as cv

# Loading a pretrained data of face frontals
face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_classifier =  cv.CascadeClassifier('smiles.xml')
eye_classifier =  cv.CascadeClassifier('eye.xml')

# Take video input from webcam
webcam = cv.VideoCapture(0)

# A While loop to iterate over each frame
while True:
    # Store frames fr om the webcam
    successful_frame_read, frame = webcam.read()
    if not successful_frame_read:
        break
    # Grayscaling frames
    grayscaled = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(grayscaled)
    # Run for loop to draw a rectangle on the face with a green rectangle and borderwidth 10 milimeters
    for (x, y, w, h) in face:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 3)
        # Slicing numpy syb-array to aquire only the face
        facial_coordinates = frame[y:y+h, x:x+w]
        # grayscaling just the face
        grayscaled_frame = cv.cvtColor(facial_coordinates, cv.COLOR_BGR2GRAY)
        # Detect smiles
        smile = smile_classifier.detectMultiScale(grayscaled_frame, scaleFactor = 1.7, minNeighbors = 20)
        # Detect eyes
        eye = eye_classifier.detectMultiScale(grayscaled_frame)
        # Detect smile coordinates within the face. Hence a nested loop
        for (X, Y, W, H) in smile:
            cv.rectangle(facial_coordinates, (X, Y), (X + W, Y + H), (0, 255, 0), 3)
            '''if len(smile) > 0:
                cv.putText(frame, 'Smirk', (x, y + h + 40), fontScale = 2,
                fontFace = cv.FONT_ITALIC, color = (0,255,0))'''
        
        for (x1, y1, w1, h1) in eye:
            cv.rectangle(facial_coordinates, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 3)

        # Run facial recognition system with colored frames
        cv.imshow('Facial Recognition System', frame)        
        cv.waitKey(1)




