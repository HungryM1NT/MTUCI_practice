from train_utils import *
import pandas as pd
import os
import numpy as np


outputFolder: str = './Pandaset'
lidarURL: str = 'https://ssd.mathworks.com/supportfiles/lidar/data/Pandaset_LidarData.tar.gz'
helperDownloadPandasetData(outputFolder, lidarURL)

path: str = f'{outputFolder}/Lidar'
# PCDS isn't iterator
pcds = np.asarray([f'{path}/{x}' for x in os.listdir(path)])


# gtPath = f'{outputFolder}/Cuboids/PandaSetLidarGroundTruth.mat'
gtPath: str = './table.csv'
df = pd.read_csv(gtPath)
boxLabels = df.drop('Time', axis=1)

# Params
xMin = -25.0
xMax = 25.0
yMin = 0.0
yMax = 50.0
zMin = -7.0
zMax = 15.0
bevHeight = 608
bevWidth = 608
gridW = (yMax - yMin)/bevWidth
gridH = (xMax - xMin)/bevHeight
gridParams = ((xMin, xMax, yMin, yMax, zMin, zMax), (bevWidth, bevHeight), (gridW, gridH))


writeFiles = True
if writeFiles:
    transformPCtoBEV(pcds, boxLabels, gridParams, outputFolder)  #TODO написать функцию


