import cv2 as cv
import numpy as np

#text output
#cv.FONT_HERSHEY_SIMPLEX : 보통 크기의 산세리프(sans-serif) 폰트
#cv.FONT_HERSHEY_PLAIN : 작은 크기의 산세리프 폰트
#cv.FONT_HERSHEY_SCRIPT_SIMPLEX : 필기체 스타일의 폰트
#cv.FONT_HERSHEY_TRIPLEX : 보통 크기의 산세리프 폰트
#cv.FONT_ITALIC : 기울임(이탤릭)체 // 이 폰트는 다른 폰트와 함께 사용해야한다.

img = np.zeros((480,640,3), np.uint8)    #create a black image

SCALE = 1   #font scale
COLOR = (255,255,255)   #font color white
THICKNESS = 1   #font thickness

cv.putText(img, 'Hello World!', (10,30), cv.FONT_HERSHEY_SIMPLEX, SCALE, COLOR, THICKNESS)
#where to put the text, text, position, font, scale, color, thickness
cv.putText(img, 'Hello World!', (10,60), cv.FONT_HERSHEY_PLAIN, SCALE, COLOR, THICKNESS)
cv.putText(img, 'Hello World!', (10,90), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, SCALE, COLOR, THICKNESS)
cv.putText(img, 'Hello World!', (10,120), cv.FONT_HERSHEY_TRIPLEX, SCALE, COLOR, THICKNESS)
cv.putText(img, 'Hello World!', (10,150), cv.FONT_HERSHEY_TRIPLEX | cv.FONT_ITALIC, SCALE, COLOR, THICKNESS)

cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()
