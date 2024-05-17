import cv2 as cv


img = cv.imread('/Users/a/Documents/GitHub/git_test_project/testimg/calibration.png', cv.IMREAD_GRAYSCALE)   # Read the image by grayscale
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()

result = cv.imwrite('/Users/a/Documents/GitHub/git_test_project/testimg/calibration_gray.png', img)     # Save the image by png format
print(result)