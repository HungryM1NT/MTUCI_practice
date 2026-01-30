import open3d as o3d
import numpy as np
from ultralytics import YOLO
from utils import *
from filler import *
import cv2 as cv
from pypcd4 import PointCloud


# def sub2ind_2D(size, row, col):
#     return (col - 1) * size[0] + row

# def matlab_hist(arr, binc):
#     gaps = [x * (max(arr) - min(arr)) / (binc - 1) for x in range(1, binc)]
#     vals = np.zeros(binc)
#     for x in arr:
#         for i in range(len(gaps)):
#             if x < gaps[i]:
#                 vals[i] += 1
#                 break
#     vals[-1] = len(arr) - sum(vals)
#     print(vals)

def preprocess(pcd_points, gridParams):
    pcdRange = gridParams[0]
    points_ROI = get_ROI_points(pcd_points, pcdRange)
    
    xMin = gridParams[0][0]
    yMin = gridParams[0][2]
    
    bevWidth = gridParams[1][0]
    bevHeight = gridParams[1][1]
    
    gridW = gridParams[2][0]
    gridH = gridParams[2][1]
    
    points_ROI[:, 0] = np.int32(np.floor((points_ROI[:, 0] - xMin) / gridH))
    points_ROI[:, 1] = np.int32(np.floor((points_ROI[:, 1] - yMin) / gridW))
    
    points_ROI[:, 2] = points_ROI[:, 2] - np.min(points_ROI[:, 2])
    points_ROI[:, 2] = points_ROI[:, 2] / (pcdRange[5] - pcdRange[4])
    
    ix = np.lexsort((-points_ROI[:, 2], points_ROI[:, 1], points_ROI[:, 0]))
    points_ROI = points_ROI[ix]
    
    heightMap = np.zeros((bevHeight, bevWidth))
    densityMap = np.zeros((bevHeight, bevWidth))
    
    points_ROI[:, 0] = np.minimum(np.maximum(points_ROI[:, 0], 0), bevHeight - 1)
    points_ROI[:, 1] = np.minimum(np.maximum(points_ROI[:, 1], 0), bevWidth - 1)
    
    coord_to_countval = get_CoordToCountVal_dict(points_ROI)
    
    for ((x, y), (c, z)) in coord_to_countval.items():
        densityMap[int(x)][int(y)] = min(1.0, np.log(c + 1) / np.log(64))
        heightMap[int(x)][int(y)] = z
    
    imageMap = np.zeros([bevHeight, bevWidth, 3])
    imageMap[:,:,0] = densityMap
    imageMap[:,:,1] = heightMap
    imageMap[:,:,2] = heightMap
    # T для переворота
    
    return imageMap
    
    
def main():
    # pcd = o3d.io.read_point_cloud("./assets/cropped.pcd")

    # points_array = np.asarray(pcd.points)
    
    # pcd = PointCloud.from_path("./assets/0001.pcd")
    pcd = PointCloud.from_path("./assets/cropped.pcd")
    points_array = pcd.numpy(("x", "y", "z"))
    points_array = points_array[~np.isnan(points_array).any(axis=1)]

    model = YOLO("./best4.pt")

    xMin = 101100.0;  xMax = 101150.0
    yMin = 85400.0;   yMax = 85450.0
    zMin = 140.0;     zMax = 220.0


    # xMin = -25.0
    # xMax = 25.0
    # yMin = 0.0
    # yMax = 50.0
    # zMin = -7.0
    # zMax = 15.0
    
    bevHeight = 608
    bevWidth  = 608

    gridW = (yMax - yMin) / bevWidth
    gridH = (xMax - xMin) / bevHeight

    gridParams = ((xMin, xMax, yMin, yMax, zMin, zMax), (bevWidth, bevHeight), (gridW, gridH))

    bevImage = preprocess(points_array, gridParams)
    bevImage = bevImage * 255
    bevImage = cv.cvtColor(bevImage.astype('float32'), cv.COLOR_RGB2BGR)

    cv.imwrite('test.jpg', bevImage)
    # img = cv.imread("./Pandaset/yolo_data/test/images/0001.jpg")
    img = cv.imread("test.jpg")

    results = model([img])
    for result in results:
        boxes = result.boxes
        result.show()
        # print(boxes)
    # bboxes = process_yolo_boxes(bevImage, model)
    # pcd_coords = bev_to_coords(bboxes, gridParams)
    # output_pcd = delete_points(points_array, pcd_coords)
    # new_outpud_pcd = fill_deleted_areas(output_pcd, bboxes)
    
    # print(len(points_array))
    # print(len(output_pcd))
    # print(len(new_outpud_pcd))
    
    
    # pcd_o = o3d.geometry.PointCloud()
    # v3d = o3d.utility.Vector3dVector
    # pcd_o.points = v3d(new_outpud_pcd)
    # o3d.io.write_point_cloud("./assets/processed.pcd", pcd_o, compressed=True)

    

if __name__ == "__main__":
    main()