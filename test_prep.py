from train_utils import preprocess
import open3d as o3d
import numpy as np
from ultralytics import YOLO
import cv2 as cv

xMin = 101100.0;  xMax = 101150.0
yMin = 85400.0;   yMax = 85450.0
zMin = 140.0;     zMax = 220.0

bevHeight = 608
bevWidth  = 608

gridW = (yMax - yMin) / bevWidth
gridH = (xMax - xMin) / bevHeight

gridParams = ((xMin, xMax, yMin, yMax, zMin, zMax), (bevWidth, bevHeight), (gridW, gridH))
    
pcd = o3d.io.read_point_cloud("./assets/cropped.pcd")

points_array = np.asarray(pcd.points)
img = preprocess(points_array, gridParams)
img = img * 255

cv.imwrite("test2.jpg", img)