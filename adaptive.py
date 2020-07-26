import cv2
img=cv2.imread('trying/test/2/newimage297.PNG',0)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

dst = cv2.fastNlMeansDenoising(th3,None,10,10,7)

cv2.imshow('new',dst)
cv2.waitKey(0)