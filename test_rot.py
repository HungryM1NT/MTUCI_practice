import numpy as np

# T AND P
# [[        238,511, 76, 26     -82.991]]
# [[        393, 55, 15, 10     -179.41]
#  [        141, 72, 11,  9     0.36869]])

def rotate_x(a):
    cos_a = np.cos(np.deg2rad(a))
    sin_a = np.sin(np.deg2rad(a))
    # print(cos_a)
    return lambda x, y: x * cos_a - y * sin_a

def rotate_y(a):
    cos_a = np.cos(np.deg2rad(a))
    sin_a = np.sin(np.deg2rad(a))
    return lambda x, y: x * sin_a + y * cos_a

#     p1 (x1, y1)=====p2(x2, y1) 
#         ||              ||
#         ||              ||
#         ||       o      ||
#         ||              ||
#         ||              ||
#     p3 (x1, y2)=====p4 (x2, y2)
def rotate_bboxes(bboxes):
    start_x1 = bboxes[:, 0] - bboxes[:, 2] / 2
    start_x2 = start_x1 + bboxes[:, 2]
    start_y1 = bboxes[:, 1] - bboxes[:, 3] / 2
    start_y2 = start_y1 + bboxes[:, 3]
    
    # Move center to (0, 0)
    start_x1 = start_x1 - bboxes[:, 0]
    start_x2 = start_x2 - bboxes[:, 0]
    start_y1 = start_y1 - bboxes[:, 1]
    start_y2 = start_y2 - bboxes[:, 1]
    
    # print(start_x1)
    rotate_x_func = rotate_x(bboxes[:, 4])
    rotate_y_func = rotate_y(bboxes[:, 4])
    # print(x_test)
    
    # x1x2x3x4y1y2y3y4
    new_corners = np.zeros((len(bboxes[:,0]), 8))
    # Make it shortier
    new_corners[:, 0] = np.round(rotate_x_func(start_x1, start_y1), 4)
    new_corners[:, 1] = np.round(rotate_x_func(start_x2, start_y1), 4)
    new_corners[:, 2] = np.round(rotate_x_func(start_x1, start_y2), 4)
    new_corners[:, 3] = np.round(rotate_x_func(start_x2, start_y2), 4)
    new_corners[:, 4] = np.round(rotate_y_func(start_x1, start_y1), 4)
    new_corners[:, 5] = np.round(rotate_y_func(start_x2, start_y1), 4)
    new_corners[:, 6] = np.round(rotate_y_func(start_x1, start_y2), 4)
    new_corners[:, 7] = np.round(rotate_y_func(start_x2, start_y2), 4)
    
    new_x_max = np.amax(new_corners[:, 0:4], axis=1)
    new_x_min = np.amin(new_corners[:, 0:4], axis=1)
    new_y_max = np.amax(new_corners[:, 5:], axis=1)
    new_y_min = np.amin(new_corners[:, 5:], axis=1)
    
    # Move center
    new_x_max = new_x_max + bboxes[:, 0]
    new_x_min = new_x_min + bboxes[:, 0]
    new_y_max = new_y_max + bboxes[:, 1]
    new_y_min = new_y_min + bboxes[:, 1]
    
    xywh = np.zeros((len(bboxes[:,0]), 4))
    xywh[:, 0] = (new_x_max + new_x_min) / 2
    xywh[:, 1] = (new_y_max + new_y_min) / 2
    xywh[:, 2] = new_x_max - new_x_min
    xywh[:, 3] = new_y_max - new_y_min
    
    return xywh
    


cars = np.asarray([
    [447, 209, 56, 22, -181.94],
    [539, 211, 57, 22, -181.94],
    [62, 167, 54, 21, -185],
    [123, 28, 50, 23, 89.374],
    [243, 419, 50, 22, -83.991],
    [329, 430, 58, 22, -63.241],
    [282, 155, 53, 21, -183],
    [525, 173, 55, 21, -181.2],
    [473, 136, 53, 21, -181],
    [197, 258, 51, 23, -0.78626]])

# cars = np.asarray([
#     [1, 6, 8, 4, 0],
#     [1, 6, 8, 4, 90],
#     [1, 6, 8, 4, 180],
#     [1, 6, 8, 4, 270],
# ])

print(rotate_bboxes(cars))