import os
import requests
import tarfile
import numpy as np
import open3d as o3d
import pandas as pd
from utils import *
import cv2 as cv
from pypcd4 import PointCloud
import re
import shutil


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
    return ctp

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
        # print(ctp)
        
        bevImage = preprocess(points_array, gridParams)
        bevImage = bevImage * 255
        
        labels_text = ''
        
        for j in range(len(classNames)):
            # print(j)
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

            csvBEV = labelsBEV.T.reshape(-1)
            processedLabels[i, OUTPUT_CSV_STARTS[classNames[j]]:OUTPUT_CSV_STARTS[classNames[j]] + len(csvBEV)] = csvBEV
            
            if labelsBEV.size != 0:
                lt = np.array2string(labelsBEV[:, :4])
                lt = re.sub("\[|\]", " ", lt)
                lt = re.sub(" +", " ", lt)
                lt = re.sub("^|\n", f"\n{j}", lt)
                lt = lt.strip()
                labels_text = "\n".join((labels_text, lt))      
    
        imagesWritePath = f'{dataLocation}/BEVImages'
        if not os.path.exists(imagesWritePath):
            os.mkdir(imagesWritePath)

        imgSavePath = f'{imagesWritePath}/{i + 1:04d}.jpg'
        img = cv.cvtColor(bevImage.astype('float32'), cv.COLOR_RGB2BGR)
        cv.imwrite(imgSavePath, img)
        
        labelsWritePath = f'{dataLocation}/Labels'
        if not os.path.exists(labelsWritePath):
            os.mkdir(labelsWritePath)
        
        labelSavePath = f'{labelsWritePath}/{i + 1:04d}.txt'
        with open(labelSavePath, 'w') as labelfile:
            labelfile.write(labels_text.strip())
        
        
    
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

def create_yolo_folder(imgs, labels, yolo_data_folder, folder_name):
    copy_folder = f"{yolo_data_folder}/{folder_name}"
    os.mkdir(copy_folder)
    os.mkdir(f"{copy_folder}/images/")
    os.mkdir(f"{copy_folder}/labels/")
    for i in range(len(imgs)):
        shutil.copy(imgs[i], f"{copy_folder}/images/")
        shutil.copy(labels[i], f"{copy_folder}/labels/")


def create_yolo_datastore(outputFolder):
    bevs_path = f'{outputFolder}/BEVImages'
    bevs = np.asarray([f'{bevs_path}/{x}' for x in os.listdir(bevs_path)])
    bevs.sort()

    labels_path = f'{outputFolder}/Labels'
    labels = np.asarray([f'{labels_path}/{x}' for x in os.listdir(labels_path)])
    labels.sort()

    rng = np.random.RandomState(42)
    indexes = rng.permutation(np.arange(bevs.size))

    train_indexes = indexes[0:int(0.6 * len(indexes))]
    validation_indexes = indexes[len(train_indexes):len(train_indexes) + int(0.1 * len(indexes))]
    test_indexes = indexes[len(validation_indexes) + len(train_indexes):]

    train_imgs = bevs[train_indexes];            train_labels = labels[train_indexes]
    validation_imgs = bevs[validation_indexes];  validation_labels = labels[validation_indexes]
    test_imgs = bevs[test_indexes];              test_labels = labels[test_indexes]
    
    if os.path.exists(f"{outputFolder}/yolo_data"):
        shutil.rmtree(f"{outputFolder}/yolo_data", )
    
    yolo_data_folder = f"{outputFolder}/yolo_data"
    os.mkdir(yolo_data_folder)
    
    create_yolo_folder(train_imgs, train_labels, yolo_data_folder, 'train')
    create_yolo_folder(validation_imgs, validation_labels, yolo_data_folder, 'validation')
    create_yolo_folder(test_imgs, test_labels, yolo_data_folder, 'test')

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

# def helperDisplayBoxes():
#     pass