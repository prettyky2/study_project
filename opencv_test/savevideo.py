#Save Video

import cv2 as cv

cap = cv.VideoCapture('/Users/a/Documents/GitHub/git_test_project/test_video/GOPR1733.MP4')     # Open the camera

#define codec, frame size, fps
fourcc = cv.VideoWriter_fourcc(*'DIVX')     # Define the codec fourcc=Four Character Code
width = round(cap.get(cv.CAP_PROP_FRAME_WIDTH))     # Get the width of the present frame(cap) and round it beacuse it is float
height = round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))     # Get the height of the present frame(cap) and round it beacuse it is float
fps = cap.get(cv.CAP_PROP_FPS)  * 2   # Get the fps of the present frame(cap) and multiply it by 2

out = cv.VideoWriter('/Users/a/Documents/GitHub/git_test_project/test_video/output_fast.mp4', fourcc, fps, (width, height))     # Define the output file
#(save path and name, codec, fps, size)

while cap.isOpened():   # Check if the camera is opened
    ret, frame = cap.read()     # Read the frame
    if not ret:     # Check if the frame is read correctly
        print('Can\'t receive frame (stream end?). Exiting ...')
        break


    out.write(frame)     # Write the frame to the output file

    cv.imshow('frame', frame)     # Show the frame
    if cv.waitKey(1) == ord('q'):     # Check if the key is pressed
        break

out.release()     # Release the output file
cap.release()     # Release the camera
cv.destroyAllWindows()     # Close the window