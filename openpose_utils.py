import cv2
import numpy as np


def draw_openpose_human_pose(keypoints, image_shape, K, RT, alpha=0.5):
    """
    Draw Openpose human pose on image
    Args:
        keypoints (numpy array): Openpose keypoints (25, 3) (index, x, y, z)
        image_shape (tuple): Image shape (height, width)
        K (numpy array): Camera intrinsic matrix (3, 3)
        RT (numpy array): Camera extrinsic matrix (3, 4)
        alpha (float): Opacity of keypoints
    """
    # Create image
    image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

    keypoints_num = len(keypoints)

    openpose_colors = []
    openpose_conn = []

    # Openpose colors, one for each keypoint(25)
    if keypoints_num == 25:
        openpose_colors = [
            (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
            (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
            (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
            (255, 0, 255), (255, 0, 170), (255, 0, 85), (255, 0, 0), (255, 85, 0),
            (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0)
        ]

        openpose_conn = [
            [0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
            [1, 8], [8, 9], [9, 10], [10, 11], [8, 12], [12, 13],
            [13, 14], [0, 15], [0, 16], [15, 17], [16, 18], [11, 24],
            [11, 22], [22, 23], [14, 21], [14, 19], [19, 20]
        ]
    elif keypoints_num == 18:
        # color for each keypoint(18)
        openpose_colors = [
            (255, 0, 85), (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
            (170, 255, 0), (85, 255, 0), (0, 255, 0), (255, 0, 0), (0, 255, 85),
            (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255),
            (255, 0, 170), (170, 0, 255), (255, 0, 255), (85, 0, 255), (0, 0, 255),
            (0, 0, 255), (0, 0, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255)
        ]

        openpose_conn = [
            [0, 1], [1, 2], [2, 3], [3, 4],
            [1, 5], [5, 6], [6, 7],
            [1, 8], [8, 9], [9, 10],
            [1, 11], [11, 12], [12, 13],
            [0, 14], [14, 16], [0, 15], [15, 17]
        ]
    else:
        raise ValueError("Invalid keypoints number")

    # Draw keypoints
    for i, keypoint in enumerate(keypoints):
        overlay = image.copy()
        # project 3D point to 2D
        point_3d = np.array([keypoint[0], keypoint[1], keypoint[2], 1])
        point_2d = K @ RT @ point_3d
        point_2d = point_2d[:2] / point_2d[2]
        # Draw circle
        cv2.circle(
            overlay, (int(point_2d[0]), int(point_2d[1])), 4, openpose_colors[i], -1
        )
        # Blend images
        image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    # Draw connections(not line, but oval)
    for connection in openpose_conn:
        num1, num2 = connection
        if num1 >= len(keypoints) or num2 >= len(keypoints):
            continue
        overlay = image.copy()
        # project 3D point to 2D
        point_3d_1 = np.array(
            [keypoints[num1][0], keypoints[num1][1], keypoints[num1][2], 1]
        )
        point_2d_1 = K @ RT @ point_3d_1
        point_2d_1 = point_2d_1[:2] / point_2d_1[2]
        point_3d_2 = np.array(
            [keypoints[num2][0], keypoints[num2][1], keypoints[num2][2], 1]
        )
        point_2d_2 = K @ RT @ point_3d_2
        point_2d_2 = point_2d_2[:2] / point_2d_2[2]
        # Draw oval
        midpoint = (point_2d_1 + point_2d_2) / 2
        angle = np.degrees(
            np.arctan2(point_2d_2[1] - point_2d_1[1], point_2d_2[0] - point_2d_1[0])
        )
        thickness = 3
        cv2.ellipse(
            overlay,
            (int(midpoint[0]), int(midpoint[1])),
            (int(np.linalg.norm(point_2d_2 - point_2d_1) / 2), thickness),
            angle,
            0,
            360,
            openpose_colors[num2],
            -1,
        )
        # Blend images
        image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    return image


def draw_openpose_human_pose_official(K, RT, keypoints=None, image_shape=(512, 512), alpha=0.5):
    """
    Draw Openpose human pose on image
    Args:
        keypoints (numpy array): Openpose keypoints (25, 3) (index, x, y, z)
        image_shape (tuple): Image shape (height, width)
        K (numpy array): Camera intrinsic matrix (3, 3)
        RT (numpy array): Camera extrinsic matrix (3, 4)
        alpha (float): Opacity of keypoints
    """
    # Create image
    image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    T_pose_keypoints = np.array(
        [
            [0, 158, 14],
            [0, 138, 0],
            [-17, 138, 0],
            [-17, 113, 0],
            [-17, 88, 0],
            [17, 138, 0],
            [17, 113, 0],
            [17, 88, 0],
            [-10, 92, 0],
            [-10, 52, 0],
            [-10, 16, 0],
            [10, 92, 0],
            [10, 52, 0],
            [10, 16, 0],
            [-3, 161, 11],
            [3, 161, 11],
            [-7, 158, 3],
            [7, 158, 3],
        ]
    )

    normalized_keypoints = mid_and_scale(T_pose_keypoints)
    if keypoints is None:
        keypoints = normalized_keypoints

    # Openpose colors, one for each keypoint(25)
    # openpose_colors = [
    #     (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    #     (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    #     (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    #     (255, 0, 255), (255, 0, 170), (255, 0, 85), (255, 0, 0), (255, 85, 0),
    #     (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0)
    # ]

    # openpose_conn = [
    #     [0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    #     [1, 8], [8, 9], [9, 10], [10, 11], [8, 12], [12, 13],
    #     [13, 14], [0, 15], [0, 16], [15, 17], [16, 18], [11, 24],
    #     [11, 22], [22, 23], [14, 21], [14, 19], [19, 20]
    # ]

    # color for each keypoint(18)
    openpose_colors = [
        (255, 0, 85),
        (255, 0, 0),
        (255, 85, 0),
        (255, 170, 0),
        (255, 255, 0),
        (170, 255, 0),
        (85, 255, 0),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 85),
        (0, 255, 170),
        (0, 255, 255),
        (0, 170, 255),
        (0, 85, 255),
        (0, 0, 255),
        (255, 0, 170),
        (170, 0, 255),
        (255, 0, 255),
        (85, 0, 255),
        (0, 0, 255),
        (0, 0, 255),
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 255),
        (0, 255, 255),
    ]

    # openpose official connection TODO: check color order
    openpose_conn = np.array(
        [1, 8, 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 8, 12, 12, 13, 13, 14, 1, 0, 0, 15, 15, 17, 0,
         16, 16, 18, 14, 19, 19, 20, 14, 21, 11, 22, 22, 23, 11, 24]).reshape(-1, 2)

    # Draw keypoints
    for i, keypoint in enumerate(keypoints):
        overlay = image.copy()
        # project 3D point to 2D
        point_3d = np.array([keypoint[0], keypoint[1], keypoint[2], 1])
        point_2d = K @ RT @ point_3d
        point_2d = point_2d[:2] / point_2d[2]
        # Draw circle
        cv2.circle(
            overlay, (int(point_2d[0]), int(point_2d[1])), 4, openpose_colors[i], -1
        )
        # Blend images
        image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    # Draw connections(not line, but oval)
    for connection in openpose_conn:
        num1, num2 = connection
        if num1 >= len(keypoints) or num2 >= len(keypoints):
            continue
        overlay = image.copy()
        # project 3D point to 2D
        point_3d_1 = np.array(
            [keypoints[num1][0], keypoints[num1][1], keypoints[num1][2], 1]
        )
        point_2d_1 = K @ RT @ point_3d_1
        point_2d_1 = point_2d_1[:2] / point_2d_1[2]
        point_3d_2 = np.array(
            [keypoints[num2][0], keypoints[num2][1], keypoints[num2][2], 1]
        )
        point_2d_2 = K @ RT @ point_3d_2
        point_2d_2 = point_2d_2[:2] / point_2d_2[2]
        # Draw oval
        midpoint = (point_2d_1 + point_2d_2) / 2
        angle = np.degrees(
            np.arctan2(point_2d_2[1] - point_2d_1[1], point_2d_2[0] - point_2d_1[0])
        )
        thickness = 3
        cv2.ellipse(
            overlay,
            (int(midpoint[0]), int(midpoint[1])),
            (int(np.linalg.norm(point_2d_2 - point_2d_1) / 2), thickness),
            angle,
            0,
            360,
            openpose_colors[num2],
            -1,
        )
        # Blend images
        image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    return image


def mid_and_scale(keypoints):
    """
    Get middle point and scale of human pose
    Args:
        keypoints (numpy array): Openpose keypoints (25, 3) (index, x, y, z)
    Returns:
        normalized_keypoints (numpy array): Normalized keypoints (25, 3) (index, x, y, z)
    """
    # Get middle point
    middle_point = np.mean(keypoints, axis=0)

    # Get scale
    scale = np.max(np.linalg.norm(keypoints[:, :2] - middle_point[:2], axis=1)) * 1.6

    # Normalize keypoints
    offset = np.array([0, 0.1, 0])
    normalized_keypoints = (keypoints - middle_point) / scale + offset
    return normalized_keypoints



# def is_back(K, RT) -> bool:
#     """
#     Check if camera is facing back
#     Args:
#         K (numpy array): Camera intrinsic matrix (3, 3)
#         RT (numpy array): Camera extrinsic matrix (3, 4)
#     Returns:
#         bool: True if camera is facing back
#     """
#     # Get camera center
#     C = -np.linalg.inv(RT[:, :3]) @ RT[:, 3]
#     # Get camera forward vector
#     forward = RT[:, 2]
#     # Get camera forward vector in image plane
#     forward = K @ forward
#     # Check if camera is facing back
#     return (C[2] < 0 and forward[2] < 0)
