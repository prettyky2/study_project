import cv2 as cv
print(cv.__version__)

#image output

#cv.IMREAD_COLOR : 이미지 파일을 Color로 읽어들입니다. 투명한 부분은 무시되며 Default값입니다.
#cv.IMREAD_GRAYSCALE : 이미지를 Grayscale로 읽어 들입니다. 실제 이미지 처리시 중간 단계로 많이 사용합니다.
#cv.IMREAD_UNCHANGED : 이미지파일을 alpha channel까지 포함하여 읽어 들입니다.

img = cv.imread('/Users/a/Documents/GitHub/git_test_project/git_test_project/testimg/calibration.png')
img2 = cv.imread('/Users/a/Documents/GitHub/git_test_project/git_test_project/testimg/calibration.png',cv.IMREAD_GRAYSCALE)
img3 = cv.imread('/Users/a/Documents/GitHub/git_test_project/git_test_project/testimg/calibration.png',cv.IMREAD_UNCHANGED)

cv.imshow('img',img)    #display the image
cv.imshow('img2',img2)
cv.imshow('img3',img3)

cv.waitKey(0)
cv.destroyAllWindows()


print(img.shape)    #image shape