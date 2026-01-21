import os
import requests
import tarfile



def helperDownloadPandasetData(outputFolder: str, lidarURL: str):
    lidarDataTarFile = f'{outputFolder}/Pandaset_LidarData.tar.gz'
    if not os.path.exists(lidarDataTarFile):
        os.mkdir(outputFolder)
        print('Downloading PandaSet Lidar driving data (5.2 GB)...')
        response = requests.get(lidarURL, stream=True)
        file = tarfile.open(fileobj=response.raw, mode="r|gz")
        file.extractall(path=outputFolder)



outputFolder = './Pandaset'
lidarURL = 'https://ssd.mathworks.com/supportfiles/lidar/data/Pandaset_LidarData.tar.gz'
helperDownloadPandasetData(outputFolder, lidarURL)

path = f'{outputFolder}/Lidar'
pass #TODO pcds

gtPath = f'{outputFolder}/Cuboids/PandaSetLidarGroundTruth.mat'




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

