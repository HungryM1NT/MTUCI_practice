import numpy as np
from ultralytics import YOLO
import open3d


# Create (x, y): (counts, maxZ) dict
def get_CoordToCountVal_dict(points):
    coord_to_countval = dict()
    for point in points:
        if coord_to_countval.get((point[0], point[1])) == None:
            coord_to_countval[((point[0], point[1]))] = [1, point[2]]
        else:
            coord_to_countval[((point[0], point[1]))][0] += 1
    return coord_to_countval


def get_ROI_mask_xyz(points, p_range):
    mask = (points[:, 0] >= p_range[0]) & \
            (points[:, 0] < p_range[1]) & \
            (points[:, 1] >= p_range[2]) & \
            (points[:, 1] < p_range[3]) & \
            (points[:, 2] >= p_range[4]) & \
            (points[:, 2] < p_range[5])
    return mask


def get_ROI_mask_xy(points, p_range):
    mask = (points[:, 0] >= p_range[0]) & \
            (points[:, 0] < p_range[1]) & \
            (points[:, 1] >= p_range[2]) & \
            (points[:, 1] < p_range[3])
    return mask


def get_delete_mask(points, p_ranges):
    mask = np.full((len(points)), False)
    for p_range in p_ranges:
        mask = np.logical_or(mask, get_ROI_mask_xy(points, p_range))
    
    return np.invert(mask)

# Collect all points with [minX, maxX), [minY, maxY), [minZ, maxZ) interval
def get_ROI_points(pcd_points, pcdRange):
    mask = get_ROI_mask_xyz(pcd_points, pcdRange)
    return pcd_points[mask]


def bev_to_coords(bev_coords, gridParams):
    bev_coords[:, 0] = bev_coords[:, 0] * gridParams[2][1] + gridParams[0][0]
    bev_coords[:, 1] = bev_coords[:, 1] * gridParams[2][1] + gridParams[0][0]
    
    bev_coords[:, 2] = bev_coords[:, 2] * gridParams[2][0] + gridParams[0][2]
    bev_coords[:, 3] = bev_coords[:, 3] * gridParams[2][0] + gridParams[0][2]
    
    return bev_coords
    

def get_yolo_boxes(yolo_results):
    bboxes = []
    for r in yolo_results:
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0]
            b = [b[0], b[2], b[1], b[3]]
            bboxes.append(b)

    return np.array(bboxes)

def process_yolo_boxes(bevImage):
    bev_coords = bevImage
    output = np.full(bev_coords.shape, 0)
    
    bev_coords[:, 2] = bev_coords[:, 2]
    bev_coords[:, 3] = bev_coords[:, 3]
    
    output[:, 0] = bev_coords[:, 1] - bev_coords[:, 3] / 2
    output[:, 1] = output[:, 0] + bev_coords[:, 3]
    output[:, 2] = bev_coords[:, 0] - bev_coords[:, 2] / 2
    output[:, 3] = output[:, 2] + bev_coords[:, 2]
        
    return output


def delete_points(pcd_points, bboxes):
    mask = get_delete_mask(pcd_points, bboxes)
    return pcd_points[mask]
