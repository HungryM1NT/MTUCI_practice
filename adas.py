import cv2 as cv


for i in range(1):
    a_my = cv.imread('./Pandaset/BEVImages/0001.jpg')
    a_matlab = cv.imread('./matlab/BEVImages/0001.jpg')
    print(f'Mine: {a_my.sum()}')
    print(f'Matlab: {a_matlab.sum()}')