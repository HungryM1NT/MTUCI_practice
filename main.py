import open3d as o3d
import numpy as np
from ultralytics import YOLO
from utils import get_ROI_points, process_yolo_boxes, bev_to_coords, delete_points


# Create (x, y): (counts, maxZ) dict
def get_CoordToCountVal_dict(points):
    coord_to_countval = dict()
    for point in points:
        if coord_to_countval.get((point[0], point[1])) == None:
            coord_to_countval[((point[0], point[1]))] = [1, point[2]]
        else:
            coord_to_countval[((point[0], point[1]))][0] += 1
    return coord_to_countval
    

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
    
    
def main():
    pcd = o3d.io.read_point_cloud("./assets/points.pcd")

    points_array = np.asarray(pcd.points)

    model = YOLO("./model/trainedYOLOv4.onnx")

    xMin = 101100.0;  xMax = 101150.0
    yMin = 85400.0;   yMax = 85450.0
    zMin = 140.0;     zMax = 220.0

    bevHeight = 608
    bevWidth  = 608

    gridW = (yMax - yMin) / bevWidth
    gridH = (xMax - xMin) / bevHeight

    gridParams = ((xMin, xMax, yMin, yMax, zMin, zMax), (bevWidth, bevHeight), (gridW, gridH))

    classNames = ('Car', 'Truck', 'Pedestrain')

    bevImage = preprocess(points_array, gridParams)

    #results = model.predict(bevImage)
    bboxes = process_yolo_boxes(gridParams)
    pcd_coords = bev_to_coords(bboxes, gridParams)
    output_pcd = delete_points(points_array, pcd_coords)
    print(len(points_array))
    print(len(output_pcd))
    
    pcd_o = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector
    pcd_o.points = v3d(output_pcd)
    o3d.io.write_point_cloud("./assets/processed.pcd", pcd_o, compressed=True)

    

if __name__ == "__main__":
    main()