import cv2
import matplotlib.pyplot as plt
import numpy as np

from gs_renderer import MiniCam
from PIL import Image

# for rendering
import pyrender
import trimesh
import pickle
import math


def draw_openpose_human_pose(
        K, RT, keypoints=None, image_shape=(512, 512), alpha=0.5, is_back=False
):
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
    neglect_points = []
    neglect_conn = []

    # Openpose colors, one for each keypoint(25)
    if keypoints_num == 25:
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

        openpose_conn = [
            [0, 1],
            [1, 2],
            [1, 5],
            [2, 3],
            [3, 4],
            [5, 6],
            [6, 7],
            [1, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [8, 12],
            [12, 13],
            [13, 14],
            [0, 15],
            [0, 16],
            [15, 17],
            [16, 18],
            [11, 24],
            [11, 22],
            [22, 23],
            [14, 21],
            [14, 19],
            [19, 20],
        ]
        if is_back:
            neglect_points = [0, 15, 16]
            neglect_conn = [[0, 1], [0, 15], [0, 16], [15, 17], [16, 18]]
    elif keypoints_num == 18:
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

        openpose_conn = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [1, 5],
            [5, 6],
            [6, 7],
            [1, 8],
            [8, 9],
            [9, 10],
            [1, 11],
            [11, 12],
            [12, 13],
            [0, 14],
            [14, 16],
            [0, 15],
            [15, 17],
        ]
        if is_back:
            neglect_points = [0, 14, 15]
            neglect_conn = [[0, 1], [0, 14], [0, 15], [14, 16], [15, 17]]
    else:
        raise ValueError("Invalid keypoints number")

    # Draw keypoints
    for i, keypoint in enumerate(keypoints):
        if i in neglect_points:
            continue
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
        if connection in neglect_conn:
            continue
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


def render_openpose(pose, cam, hor, T_pose_keypoints=None):
    w2c = np.linalg.inv(pose)
    w2c[1:3, :3] *= -1
    w2c[:3, 3] *= -1
    K = cam.K()
    RT = w2c[:3, :]

    if abs(hor) > 120:
        is_back = True
    else:
        is_back = False

    openpose_image = draw_openpose_human_pose(
        K,
        RT,
        keypoints=T_pose_keypoints,
        is_back=is_back,
    )

    return Image.fromarray(openpose_image)


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


class OpenposeRenderer:

    def __init__(self, keypoints_path, mesh_path):

        self.keypoints = pickle.load(open(keypoints_path, 'rb'))
        mesh = trimesh.load(mesh_path)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene()
        # add mesh node
        scene.add(mesh)
        fovy = 49.1
        K = np.zeros((3, 3))
        K[0, 0] = 512 / math.tan(np.deg2rad(fovy) / 2)
        K[1, 1] = 512 / math.tan(np.deg2rad(fovy) / 2)
        K[0, 2] = 512 / 2
        K[1, 2] = 512 / 2
        K[2, 2] = 1
        camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.01, zfar=10000.0)
        self.scene = scene
        self.camera = camera
        print("Only suppot fovy=49.1 now")

    def render(self, pose, cam, hor, need_depth=True):
        w2c = np.linalg.inv(pose)
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1
        K = cam.K()
        RT = w2c[:3, :]

        if abs(hor) > 120:
            is_back = True
        else:
            is_back = False

        openpose_image = draw_openpose_human_pose(
            K,
            RT,
            keypoints=self.keypoints,
            is_back=is_back,
        )
        openpose_image = Image.fromarray(openpose_image)
        if not need_depth:
            return openpose_image
        self.scene.add(self.camera, pose=pose)
        r = pyrender.OffscreenRenderer(512, 512, point_size=2)
        _, depth = r.render(self.scene)
        depth = depth / np.max(depth)
        depth[depth > 0] = 1 - depth[depth > 0]
        depth = depth / np.max(depth)

        depth = depth * 255
        depth = depth.astype(np.uint8)
        # gray to rgb
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)

        depth = Image.fromarray(depth.astype(np.uint8))

        camera_node = list(self.scene.get_nodes(obj=self.camera))[0]
        self.scene.remove_node(camera_node)

        return openpose_image, depth
