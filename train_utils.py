from sympy.core.tests.test_priority import l
import os
import requests
import tarfile
import numpy as np
import open3d as o3d
import pandas as pd
from utils import *


CSV_CONSTS = {
    "Car": 666,
    "Truck": 99,
    "Pedestrain": 621
}


# Pandaset download
def helperDownloadPandasetData(outputFolder: str, lidarURL: str):
    lidarDataTarFile = f'{outputFolder}/Pandaset_LidarData.tar.gz'
    if not os.path.exists(lidarDataTarFile):
        os.mkdir(outputFolder)
        print('Downloading PandaSet Lidar driving data (5.2 GB)...')
        response = requests.get(lidarURL, stream=True)
        file = tarfile.open(fileobj=response.raw, mode="r|gz")
        file.extractall(path=outputFolder)

def preprocessData():
    pass

def helperDisplayBoxes():
    pass

def transformPCtoBEV(lidarData, boxLabels: pd.DataFrame, gridParams, outputFolder):
    classNames = ["Car", "Truck", "Pedestrain"]
    numFiles = boxLabels.shape[0]
    processedLabels = np.zeros(boxLabels.shape)
    
    # for i in range(numFiles):
    #     pass
        
        
    i = 1
    pcd_filename = lidarData[i]
    pcd = o3d.io.read_point_cloud(pcd_filename)  # ty:ignore[possibly-missing-attribute]
    points_array = np.asarray(pcd.points)
    
    groundTruth = boxLabels.iloc[i].values
    
    bevImage = preprocess(points_array, gridParams)
    
    # for j in range(len(classNames))
    #    pass
    j = 0
    # Variable for table in csv format (Car_1..Car_2 it's only one car)
    k = 0
    nulls = np.argwhere(pd.isnull(groundTruth))
    c_num = np.int32(np.min(nulls) / 9)
    t_num = np.int32((np.min(nulls[np.where(nulls >= CSV_CONSTS['Car'])]) - CSV_CONSTS['Car']) / 9)
    p_num = np.int32((np.min(nulls[np.where(nulls >= CSV_CONSTS['Car'] + CSV_CONSTS['Truck'])])
                      - CSV_CONSTS['Car'] - CSV_CONSTS['Truck']) / 9)

    print(c_num)
    print(t_num)
    print(p_num)
    while pd.notnull(groundTruth[k]):
        label = groundTruth[k:k + 9]
        k += 9
        # print(label) 
        
        # label_Indices = 
    
    # print(groundTruth[666+t_num: 666+t_num * 2]) 
        
    # print(k) 
    # labels = 
    

# Similar to main preprocess
def preprocess(pcd_points, gridParams):
    pcdRange = gridParams[0]
    points_ROI = get_ROI_points(pcd_points, pcdRange)
    
    xMin = gridParams[0][0]
    yMin = gridParams[0][2]
    
    bevWidth = gridParams[1][0]
    bevHeight = gridParams[1][1]
    
    gridW = gridParams[2][0]
    gridH = gridParams[2][1]
    
    # Here are differences
    points_ROI[:, 0] = np.int32(np.floor(points_ROI[:, 0] / gridH) + bevHeight / 2) + 1
    points_ROI[:, 1] = np.int32(np.floor(points_ROI[:, 1] / gridW)) + 1
    
    points_ROI[:, 2] = points_ROI[:, 2] - np.min(points_ROI[:, 2])
    points_ROI[:, 2] = points_ROI[:, 2] / (pcdRange[5] - pcdRange[4])
    
    ix = np.lexsort((points_ROI[:, 2][::-1], points_ROI[:, 1], points_ROI[:, 0]))
    points_ROI = points_ROI[ix]
    
    heightMap = np.zeros((bevHeight, bevWidth))
    densityMap = np.zeros((bevHeight, bevWidth))
    
    points_ROI[:, 0] = np.minimum(np.maximum(points_ROI[:, 0], 0), bevHeight)
    points_ROI[:, 1] = np.minimum(np.maximum(points_ROI[:, 1], 0), bevWidth)
    
    coord_to_countval = get_CoordToCountVal_dict(points_ROI)
    
    for ((x, y), (c, z)) in coord_to_countval.items():
        densityMap[int(x)][int(y)] = min(1.0, np.log(c + 1) / np.log(64))
        heightMap[int(x)][int(y)] = z
    
    imageMap = np.zeros([bevHeight, bevWidth, 3])
    imageMap[:,:,0] = densityMap
    imageMap[:,:,1] = heightMap
    imageMap[:,:,2] = heightMap
    
    return imageMap

def removeEmptyData():
    pass

def validateInputDataComplexYOLOv4():
    pass

def isValidDetectorData():
    pass

def iCheckImages():
    pass

def iCheckBoxes():
    pass

def iCheckLabels():
    pass