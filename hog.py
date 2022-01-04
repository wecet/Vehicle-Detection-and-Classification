import cv2

hog = cv2.HOGDescriptor()
im = cv2.imread('test.png')
h = hog.compute(im)
cv2.imshow('Test', h)
cv2.waitKey(0)