import numpy as np
from utils import *


def equation_plane(plane_ponts):
    point1 = plane_ponts[0]
    point2 = plane_ponts[1]
    point3 = plane_ponts[2]
    
    a1 = point2[0] - point1[0]
    b1 = point2[1] - point1[1]
    c1 = point2[2] - point1[2]
    
    a2 = point3[0] - point1[0]
    b2 = point3[1] - point1[1]
    c2 = point3[2] - point1[2]
    
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - a2 * b1
    d = (-a * point1[0] - b * point1[1] - c * point1[2])
    return [a, b, c, d]


# def get_fill_points(xy_coords, plane_params):
#     z_coords = -(plane_params[0] * xy_coords[:, 0] + plane_params[1] * xy_coords[:, 1] + plane_params[3]) / plane_params[2]
#     xyz_coords = np.append(xy_coords, z_coords.reshape(-1, 1), axis=1)
#     print(xyz_coords)
    
    
def get_plane_points_with_density(pcd_points, bboxes):
    densities = []
    plane_points = []
    for bbox in bboxes:
        if bbox[3] - bbox[2] < bbox[1] - bbox[0]:
            temp = bbox[0]
            bbox[0] = bbox[1], bbox[1] = bbox[2], bbox[2] = bbox[3], bbox[3] = temp
        
        d_lower = bbox[1] - bbox[0]
        d_higher = bbox[3] - bbox[2]
        
        areas = [[bbox[0], bbox[1], bbox[3], bbox[3] + d_higher * 1.05],    # *Up
                 [bbox[0], bbox[1], bbox[2] - d_higher * 1.05, bbox[2]],    # *Down
                 [bbox[0] - d_lower * 1.05, bbox[0], bbox[2], bbox[3]],     # *Left
                 [bbox[1], bbox[1] + d_higher * 1.05, bbox[2], bbox[3]]     # *Right
                 ]
        
        point_list1 = pcd_points[get_ROI_mask_xy(pcd_points, areas[0])]
        point_list2 = pcd_points[get_ROI_mask_xy(pcd_points, areas[1])]
        point_list3_1 = pcd_points[get_ROI_mask_xy(pcd_points, areas[2])]
        point_list3_2 = pcd_points[get_ROI_mask_xy(pcd_points, areas[3])]
        point_list3 = np.concatenate((point_list3_1, point_list3_2))
        
        mean_1 = np.mean(point_list1, axis=0)
        mean_2 = np.mean(point_list2, axis=0)
        mean_3 = np.mean(point_list3, axis=0)
        
        s = (areas[0][1] - areas[0][0]) * (areas[0][3] - areas[0][2]) + \
            (areas[1][1] - areas[1][0]) * (areas[1][3] - areas[1][2]) + \
            (areas[2][1] - areas[2][0]) * (areas[2][3] - areas[2][2]) + \
            (areas[3][1] - areas[3][0]) * (areas[3][3] - areas[3][2])

        counts = len(point_list1) + len(point_list2) + len(point_list3)

        densities.append(counts / s)
        
        plane_points.append([mean_1, mean_2, mean_3])
    
    return [np.asarray(plane_points), np.asarray(densities)]


def get_plane_params(plane_points):
    plane_params = []
    for pp in plane_points:
        plane_params.append(equation_plane(pp))

    return np.asarray(plane_params)
    
    
def get_filling_points(plane_params, bboxes, densities):
    dxy = np.full((len(bboxes), 2), 0)
    dxy[:, 0] = bboxes[:, 1] - bboxes[:, 0]
    dxy[:, 1] = bboxes[:, 3] - bboxes[:, 2]
    points_count =  dxy[:, 0] * dxy[:, 1] * densities
    
    filling_points = []
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        current_plane_params = plane_params[i]
        
        points_in_row = int(points_count[i] ** 0.5) + 1
        xs = np.linspace(bbox[0], bbox[1], points_in_row)
        ys = np.linspace(bbox[2], bbox[3], points_in_row)
        for x in xs:
            for y in ys:
                z = -(current_plane_params[0] * x + current_plane_params[1] * y + current_plane_params[3]) / current_plane_params[2]
                filling_points.append([x, y, z])
    # z_coords = -(plane_params[0] * xy_coords[:, 0] + plane_params[1] * xy_coords[:, 1] + plane_params[3]) / plane_params[2]
    # xyz_coords = np.append(xy_coords, z_coords.reshape(-1, 1), axis=1)
    # print(xyz_coords)
    # print(plane_params)
    # x = np.linspace(0, 100, 10)
    
    return np.asarray(filling_points)
    

def fill_deleted_areas(pcd_points, bboxes):
    [plane_points, densities] = get_plane_points_with_density(pcd_points, bboxes)
    plane_params = get_plane_params(plane_points)

    filling_points = get_filling_points(plane_params, bboxes, densities)
    pcd_points = np.concatenate((pcd_points, filling_points))
    return pcd_points
    # new_points = np.array(())
    
    
    # print(plane_params)

# point1 = [1, 1, 3]
# point2 = [3, 4, 7]
# point3 = [-2, 2, -14]

# # point1 = [1, 1, -9]
# # point2 = [3, 4, -26]
# # point3 = [-2, 2, -16.5]

# pl_param = equation_plane(point1, point2, point3)
# # print(pl_param)

# arr = np.array(((2, 2), (-4, 3), (5, 5)))

# # z = ge
# get_fill_points(arr, pl_param)

# dx = 100
# dy = 100
# s1 = dx * dy

# num_points = 2000

# dens = num_points / s1

# dfx = 50
# dfy = 50
# sf = dfx * dfy

# point_n = dens * sf
# points_in_line = int(point_n ** 0.5) + 1
# # print(points_in_line)

# # print(np.arange(0, 100, 100 / 10))
# print(np.linspace(0, 100, 10))