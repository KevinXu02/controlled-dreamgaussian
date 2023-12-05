# Copyright (c) Facebook, Inc. and its affiliates.
import math
import os
import sys
import os.path as osp
import time

import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json
import pickle
from datetime import datetime
import pickle as pkl

from frankmocap.demo.demo_options import DemoOptions
from frankmocap.bodymocap.body_mocap_api import BodyMocap
from frankmocap.bodymocap.body_bbox_detector import BodyPoseEstimator
import frankmocap.mocap_utils.demo_utils as demo_utils
import frankmocap.mocap_utils.general_utils as gnu
from frankmocap.mocap_utils.timer import Timer

import frankmocap.renderer.image_utils as imu
from frankmocap.renderer.viewer2D import ImShow

from scipy import sparse


def run_body_mocap(body_bbox_detector, body_mocap, visualizer, image_path, out_dir=None):
    # Setup input data to handle different types of inputs

    timer = Timer()

    img_original_bgr = image_path

    body_pose_list, body_bbox_list = body_bbox_detector.detect_body_pose(
        img_original_bgr)

    if len(body_bbox_list) < 1:
        print(f"No body deteced!")
        return

        # Sort the bbox using bbox size
    # (to make the order as consistent as possible without tracking)
    bbox_size = [(x[2] * x[3]) for x in body_bbox_list]
    idx_big2small = np.argsort(bbox_size)[::-1]
    body_bbox_list = [body_bbox_list[i] for i in idx_big2small]
    body_bbox_list = [body_bbox_list[0], ]

    # Body Pose Regression
    pred_output_list = body_mocap.regress(img_original_bgr, body_bbox_list)
    assert len(body_bbox_list) == len(pred_output_list)

    # extract mesh for rendering (vertices in image space and faces) from pred_output_list
    pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)
    pred_mesh_list[0]['vertices'] = pred_mesh_list[0]['vertices'] - np.mean(pred_mesh_list[0]['vertices'], axis=0,
                                                                            keepdims=True)

    timer.toc(bPrint=True, title="Time")

    # return openpose body25 keypoints
    return pred_output_list[0]['pred_joints_3d'][:25], pred_output_list[0]['pred_vertices_smpl'], pred_mesh_list[0]['faces']


def mid_and_scale_kpts_mesh(keypoints, mesh):
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
    normalized_mesh = (mesh - middle_point) / scale + offset
    return normalized_keypoints, normalized_mesh


def image2keypoint(image: np.ndarray, out_dir=None):
    """
    Generate keypoints from image. First using frankmocap to get body mesh, then using body25 regressor get keypoints.
    Args:
        image: read from cv2.imread, BGR format
        out_dir: directory to save mesh.obj. leave None if not save

    Returns:
        np.array, shape=(25, 3), body25 keypoints of centered mesh
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    # Set bbox detector
    body_bbox_detector = BodyPoseEstimator()

    # Set mocap regressor
    use_smplx = False
    checkpoint_path = './frankmocap/extra_data/body_module/pretrained_weights/2020_05_31-00_50_43-best-51.749683916568756.pt'

    body_mocap = BodyMocap(checkpoint_path, "./frankmocap/extra_data/smpl/", device, use_smplx)

    smpl, vertices, faces = run_body_mocap(body_bbox_detector, body_mocap, None, image, out_dir)

    # body25 -> body18
    smpl = smpl[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11,
                 12, 13, 14, 15, 16, 17, 18], :]

    smpl, vertices = mid_and_scale_kpts_mesh(smpl, vertices)

    if out_dir:
        gnu.save_mesh_to_obj(os.path.join(out_dir, 'mesh.obj'), vertices, faces)
        # save smpl as pickle
        with open(os.path.join(out_dir, 'openpose.pkl'), 'wb') as f:
            pkl.dump(smpl, f)
    return smpl






if __name__ == '__main__':
    img = cv2.imread('frankmocap/sample_data/IMG_0871.JPG')
    out_dir = 'frankmocap/output'
    image2keypoint(img, out_dir)

    from cam_utils import orbit_camera
    from openpose_utils import *
    import cv2
    import numpy as np

    fovy = 49.1
    pose = orbit_camera(-0, 0, 3.5)
    w2c = np.linalg.inv(pose)
    w2c[1:3, :3] *= -1
    w2c[:3, 3] *= -1
    # print(pose)

    print("fovy:", np.deg2rad(fovy))
    K = np.zeros((3, 3))
    K[0, 0] = 512 / math.tan(np.deg2rad(fovy) / 2)
    K[1, 1] = 512 / math.tan(np.deg2rad(fovy) / 2)
    K[0, 2] = 512 / 2
    K[1, 2] = 512 / 2
    K[2, 2] = 1
    RT = w2c[:3, :]
    # focal_length = 500
    # fx = focal_length * image_shape[1] / image_shape[0]
    # fy = focal_length
    # cx, cy = image_shape[1] / 2, image_shape[0] / 2  # 光心
    # K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    import matplotlib.pyplot as plt
    import trimesh, pyrender

    normalized_keypoints = pickle.load(open('frankmocap/output/openpose.pkl', 'rb'))
    # load obj
    mesh = trimesh.load('frankmocap/output/mesh.obj')

    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    # add mesh node
    scene.add(mesh)


    # add camera node
    camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.01, zfar=1000.0)


    for i in range(0,360,36):
        pose = orbit_camera(-0, i, 3.5)
        w2c = np.linalg.inv(pose)
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1
        RT = w2c[:3, :]
        result = draw_openpose_human_pose(K, RT, normalized_keypoints, (512, 512), )
        plt.imshow(result)
        plt.show()
        time.sleep(0.1)

        camera_pose = pose
        scene.add(camera, pose=camera_pose)
        # add light node
        light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi / 16.0)
        scene.add(light, pose=camera_pose)

        # render
        r = pyrender.OffscreenRenderer(512, 512, point_size=2)
        color, depth = r.render(scene)
        # normalize depth to 0-255
        depth = depth / np.max(depth) * 255
        depth = 255 - depth
        depth[depth == 255] = 0
        plt.imshow(color)
        plt.title("color: " + str(i))
        plt.show()
        time.sleep(0.1)

        plt.imshow(depth, cmap=plt.cm.gray)
        plt.title("depth: " + str(i))
        plt.show()
        time.sleep(0.1)
        # remove camera and light
        camera_node = list(scene.get_nodes(obj=camera))[0]
        scene.remove_node(camera_node)
        light_node = list(scene.get_nodes(obj=light))[0]
        scene.remove_node(light_node)
