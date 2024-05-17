import cv2 as cv
cap = cv.VideoCapture('/Users/a/Desktop/RawData/240317/GOPR1734.MP4')

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv.imshow('frame', frame)
    
    if cv.waitKey(1) == ord('q'):   #wait for 1ms and check if 'q' is pressed
        print("q pressed. Exiting ...")
        break


cap.release()
cv.destroyAllWindows()