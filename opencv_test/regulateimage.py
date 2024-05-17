#보간법
#보간법은 이미지의 크기를 변경할 때 픽셀 사이의 값을 조절하는 방법이다.
#cv.INTER_AREA         # 크기를 줄일 때 사용
#cv.INTER_CUBIC        # 크기를 늘릴 때 사용(속도 느림, 퀄리티 좋음)
#cv.INTER_LINEAR       # 크기를 늘릴 때 사용(기본값)
#cv.INTER_NEAREST     # Nearest-neighbor interpolation

import cv2 as cv

img = cv.imread('/Users/a/Documents/GitHub/git_test_project/test_image/calibration.png')
dst = cv.resize(img, (400, 500))     # Resize the image fixed size
dst2 = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)     # Resize the image 0.5 ratio size
dst3 = cv.resize(img, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)     # Resize the image 1.5 ratio size

cv.imshow('image', img)
cv.imshow('resize', dst)
cv.imshow('resize2', dst2)
cv.imshow('resize3', dst3)

cv.waitKey(0)
cv.destroyAllWindows()     # Close the window


