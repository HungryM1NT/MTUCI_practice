import numpy as np
import cv2
import scipy.io

def imageMapTest():
    A = np.array(((2, 1, 0.4),
                (1, 2, 0.5),
                (3, 0, 0.34),
                (1, 2, 0.3),
                (0, 0, 0.15),
                (1, 0, 0.78)))

    coord_to_countval = dict()
    for i in A:
        if coord_to_countval.get((i[0], i[1])) == None:
            coord_to_countval[(i[0], i[1])] = [1, i[2]]
        else:
            coord_to_countval[(i[0], i[1])][0] += 1
            
    densityMap = np.zeros([4, 3])
    heightMap = np.zeros([4, 3])

    for ((x, y), (c, v)) in coord_to_countval.items():
        densityMap[int(x)][int(y)] = min(1.0, np.log(c + 1) / np.log(64))
        heightMap[int(x)][int(y)] = v


    imageMap = np.zeros([4, 3, 3])
    imageMap[:,:,0] = densityMap
    imageMap[:,:,1] = heightMap
    imageMap[:,:,2] = heightMap

    print(imageMap.shape)


# mat_data = scipy.io.loadmat("./model/trainedYOLOv4.mat")
# print(mat_data.keys())
# w = mat_data["None"]
# for i in w[0]:
#     print(i)
# print(w)



mat_data = scipy.io.loadmat("./PandaSetLidarGroundTruth.mat")
print(mat_data)