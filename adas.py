import cv2 as cv


for i in range(10):
    a = cv.imread('./Pandaset/BEVImages/8.jpg')
    print(a.sum())