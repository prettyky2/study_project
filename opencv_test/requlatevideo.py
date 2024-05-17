import cv2 as cv

cap = cv.VideoCapture('/Users/a/Documents/GitHub/git_test_project/test_video/GOPR1733.MP4')     # Open the camera

while cap.isOpened():
    ret, frame = cap.read()     # Read the camera frame
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    frame_resized_fixed = cv.resize(frame, (400, 500))     # Resize the frame fixed size
    frame_resized_ratio = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)     # Resize the frame 0.5 ratio size

    cv.imshow('frame', frame)     # Show the frame
    cv.imshow('frame_resized_fixed', frame_resized_fixed)     # Show the resized frame
    cv.imshow('frame_resized_ratio', frame_resized_ratio)     # Show the resized frame
    
    if cv.waitKey(1) == ord('q'):     # If you press 'q', the window will be closed
        break

cap.release()     # Release the camera
cv.destroyAllWindows()     # Close the window
    