import os
import requests
import tarfile
import numpy as np
import open3d as o3d
import pandas as pd
from utils import *
import cv2 as cv
from pypcd4 import PointCloud


MAX_ELEMENTS = {
    "Car": 74,
    "Truck": 11,
    "Pedestrain": 69 

}

INPUT_CSV_STARTS = {
    "Car": 0,
    "Truck": 74 * 9,
    "Pedestrain": 74 * 9 + 11 * 9,
}

OUTPUT_CSV_STARTS = {
    "Car": 0,
    "Truck": 74 * 5,
    "Pedestrain": 74 * 5 + 11 * 5,
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

def get_ctp(groundTruth):
    nulls = np.argwhere(pd.isnull(groundTruth))

    if len(nulls):
        c_num = np.int32(min(np.min(nulls) / 9, MAX_ELEMENTS['Car']))
    else:
        c_num = MAX_ELEMENTS['Car']
        
    if len(nulls[np.where(nulls >= INPUT_CSV_STARTS['Truck'])]):
        t_num = np.int32(min((np.min(nulls[np.where(nulls >= INPUT_CSV_STARTS['Truck'])])
                            - INPUT_CSV_STARTS['Truck']) / 9, MAX_ELEMENTS['Truck']))
    else:
        t_num = MAX_ELEMENTS['Truck']
    
    if len(nulls[np.where(nulls >= INPUT_CSV_STARTS['Pedestrain'])]):
        p_num = np.int32(min((np.min(nulls[np.where(nulls >= INPUT_CSV_STARTS['Pedestrain'])])
                        - INPUT_CSV_STARTS['Pedestrain']) / 9, MAX_ELEMENTS['Pedestrain']))
    else:
        p_num = MAX_ELEMENTS['Pedestrain']

    ctp = [c_num, t_num, p_num]

def transformPCtoBEV(lidarData, boxLabels: pd.DataFrame, gridParams, dataLocation):
    classNames = ["Car", "Truck", "Pedestrain"]
    
    cars = [f'Car_{i + 1}' for i in range(74 * 5)]
    trains = [f'Train_{i + 1}' for i in range(11 * 5)]
    pedestrians = [f'Pedestrain_{i + 1}' for i in range(69 * 5)]
    feature_list = np.concatenate((cars, trains, pedestrians))
    
    numFiles = boxLabels.shape[0]
    processedLabels = np.empty((numFiles, len(feature_list)))
    processedLabels[:] = np.nan
    
    
    for i in range(numFiles):
        pcd_filename = lidarData[i]
        pcd = PointCloud.from_path(pcd_filename)
        points_array = pcd.numpy(("x", "y", "z"))
        points_array = points_array[~np.isnan(points_array).any(axis=1)]
                  
        groundTruth = boxLabels.iloc[i].values
        ctp = get_ctp(groundTruth)
        
        bevImage = preprocess(points_array, gridParams)
        bevImage = bevImage * 255
        
        for j in range(len(classNames)):
            # In csv format (Car_1..9 it's only one car)
            class_num = ctp[j]
            labels = groundTruth[INPUT_CSV_STARTS[classNames[j]]:INPUT_CSV_STARTS[classNames[j]] + class_num * 9].reshape(9, class_num).T

            labelsIndices = ((labels[:, 0] - labels[:, 3]) > gridParams[0][0]) \
                        & ((labels[:, 0] + labels[:, 3]) < gridParams[0][1]) \
                        & ((labels[:, 1] - labels[:, 4]) > gridParams[0][2]) \
                        & ((labels[:, 1] + labels[:, 4]) < gridParams[0][3]) \
                        & ((labels[:, 3] > 0)) \
                        & ((labels[:, 4] > 0)) \
                        & ((labels[:, 5] > 0))

            labels = labels[labelsIndices]
            
            labelsBEV = labels[:,[1, 0, 4, 3, 8]]
            labelsBEV[:, 4] = -labelsBEV[:, 4]
            
            labelsBEV[:, 0] = np.int32(np.floor(labelsBEV[:, 0] / gridParams[2][0]))
            labelsBEV[:, 1] = np.int32(np.floor(labelsBEV[:, 1] / gridParams[2][1]) + gridParams[1][1] / 2)
            
            labelsBEV[:, 2] = np.int32(np.floor(labelsBEV[:, 2] / gridParams[2][0]))
            labelsBEV[:, 3] = np.int32(np.floor(labelsBEV[:, 3] / gridParams[2][1]))

            labelsBEV = labelsBEV.T.reshape(-1)
            processedLabels[i, OUTPUT_CSV_STARTS[classNames[j]]:OUTPUT_CSV_STARTS[classNames[j]] + len(labelsBEV)] = labelsBEV
            
    
        writePath = f'{dataLocation}/BEVImages'
        if not os.path.exists(writePath):
            os.mkdir(writePath)

        imgSavePath = f'{writePath}/{i + 1:04d}.jpg'
        img = cv.cvtColor(bevImage.astype('float32'), cv.COLOR_RGB2BGR)
        cv.imwrite(imgSavePath, img)
        
    
    processedLabels_df = pd.DataFrame(processedLabels, columns=feature_list)
    processedLabels_df.to_csv(f"{dataLocation}/Cuboids/BEVGroundTruthLabels.csv")
    

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
    
    points_ROI[:, 0] = np.int32(np.floor(points_ROI[:, 0] / gridH) + bevHeight / 2)
    points_ROI[:, 1] = np.int32(np.floor(points_ROI[:, 1] / gridW))
    
    points_ROI[:, 2] = points_ROI[:, 2] - np.min(points_ROI[:, 2])
    points_ROI[:, 2] = points_ROI[:, 2] / (pcdRange[5] - pcdRange[4])
    
    ix = np.lexsort((-points_ROI[:, 2], points_ROI[:, 1], points_ROI[:, 0]))
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

def preprocessData():
    pass

def helperDisplayBoxes():
    pass