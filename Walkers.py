import cv2


# Create our body classifier
body_classifier = cv2.CascadeClassifier('haarcascade_fulllbody.xml')


# Initiate video capture for video file
cap = cv2.VideoCapture('walking1.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
    # Pass frame to our body classifier
    
    bodies= body_classifier.detectMultiScale(gray,1.2,3)
    print(bodies)

    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in bodies:
       cv2.rectangle(cap,(x,y),(x+h,y+w),(40,150,150),5)
             
    cv2.imshow('cap',cap)
    if cv2.waitKey(25)==32:
       break

cap.release()
cv2.destroyAllWindows()
