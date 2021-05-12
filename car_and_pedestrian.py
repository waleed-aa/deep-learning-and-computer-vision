# Importing opencv
import cv2 as cv

# Reads the pixels from the img file and stores it in the form of vectors.
video = cv.VideoCapture('Videos/dashcam.mp4')

# Assigning pretrained ML file
car_xml = 'cars.xml'
pedestrian_xml = 'pedestrain_detection_model.xml'


# Initiate car and pedestrian HaarCascade classifier using pretrained model
car_tracking_system = cv.CascadeClassifier(car_xml)
predestrian_tracker = cv.CascadeClassifier(pedestrian_xml)


# This while loop iterates over each frame.
while True:
    (read_frame_successful, frame) = video.read()
    # only proceed if frame reading is successful
    if read_frame_successful:
        # Transform image from color to grayscale
        grayscaled_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        break    

    # Track cars and pedestrians    
    car = car_tracking_system.detectMultiScale(grayscaled_frame)
    pedestrian = predestrian_tracker.detectMultiScale(grayscaled_frame)

    # Running a for loop to create a rectangle around the cars using pixel coordinates.
    for (x,y,w,h) in car:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 1)
    
    for (x,y,w,h) in pedestrian:
        cv.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 1)


    # Initiate tracking system using colored frames.
    cv.imshow('Car Tracking System', frame)
    key = cv.waitKey(1)
    if key == '115' or key =='83':
        break

