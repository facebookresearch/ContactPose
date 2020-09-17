# Copyright (c) Facebook, Inc. and its affiliates.
# Code by Samarth Brahmbhatt
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import trimesh
import pyrender
import numpy as np
import transforms3d.euler as txe
import utilities.misc as mutils
import cv2

osp = os.path


class DepthRenderer(object):
  """
  Renders object or hand mesh into a depth map
  """
  def __init__(self, object_name_or_mesh, K, camera_name, mesh_scale=1.0):
    """
    object_name_or_mesh: either object name string (for objects),
    or {'vertices': ..., 'faces': ...} (for hand mesh)
    K: 3x3 intrinsics matrix 
    mesh_scale: scale factor applied to the mesh (1.0 for hand, 1e-3 for object)
    """
    self.K = K
    self.camera_name = camera_name
    if camera_name == 'kinect2_middle':
        self.flip_fn = lambda x: cv2.flip(cv2.flip(x, 0), 1)
        self.out_imsize = (960, 540)
    elif camera_name == 'kinect2_left':
        self.flip_fn = lambda x: cv2.flip(cv2.transpose(x), 1)
        self.out_imsize = (540, 960)
    elif camera_name == 'kinect2_right':
        self.flip_fn = lambda x: cv2.flip(cv2.transpose(x), 0)
        self.out_imsize = (540, 960)
    else:
        raise NotImplementedError
    
    # mesh
    if isinstance(object_name_or_mesh, str):
      filename = osp.join('data', 'object_models',
                          '{:s}.ply'.format(object_name_or_mesh))
      mesh_t = trimesh.load_mesh(filename)
    elif isinstance(object_name_or_mesh, dict):
      mesh_t = trimesh.Trimesh(vertices=object_name_or_mesh['vertices'],
                               faces=object_name_or_mesh['faces'])
    else:
      raise NotImplementedError
    mesh_t.apply_transform(np.diag([mesh_scale, mesh_scale, mesh_scale, 1]))
    self.oX = mesh_t.vertices
    mesh = pyrender.Mesh.from_trimesh(mesh_t)
    
    self.scene = pyrender.Scene()
    self.scene.add(mesh, pose=np.eye(4))

    # camera
    camera = pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                                       znear=0.1, zfar=2.0)
    self.camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    self.scene.add_node(self.camera_node)
    self.cTopengl = np.eye(4)
    self.cTopengl[:3, :3] = txe.euler2mat(np.pi, 0, 0)

    # renderer object
    self.renderer = pyrender.OffscreenRenderer(960, 540)

  
  def render(self, object_pose):
    """
    returns depth map produced by rendering the mesh
    object_pose: 4x4 pose of object w.r.t. camera, from ContactPose.object_pose()
    object_pose = cTo in the naming convention
    """
    oTc = np.linalg.inv(object_pose)
    oTopengl = oTc @ self.cTopengl
    self.scene.set_pose(self.camera_node, oTopengl)
    # TODO: figure out DEPTH_ONLY rendering mode with OSMesa backend
    # DEPTH_ONLY + OSMesa does not work currently
    # so we have to render color also :(
    _, depth = self.renderer.render(self.scene)
    return self.flip_fn(depth)

  
  def object_visibility_and_projections(self, object_pose, depth_thresh=5e-3):
    """
    returns projection locations of object mesh vertices (Nx2)
    and their binary visibility from the object_pose
    object_pose = cTo 4x4 pose of object w.r.t. camera
    This is cheap Z-buffering. We use rendered depth maps because they are
    cleaner than Kinect depth maps
    """
    # render depth image
    depth_im = self.render(object_pose)
    
    # project all vertices
    cX = mutils.tform_points(object_pose, self.oX)
    P = mutils.get_A(self.camera_name) @ self.K @ np.eye(4)[:3]
    cx = mutils.project(P, cX)

    # determine visibility
    visible = cX[:, 2] > 0
    visible = np.logical_and(visible, cx[:, 0] >= 0)
    visible = np.logical_and(visible, cx[:, 1] >= 0)
    visible = np.logical_and(visible, cx[:, 0] <  self.out_imsize[0]-1)
    visible = np.logical_and(visible, cx[:, 1] <  self.out_imsize[1]-1)
    u = np.round(cx[:, 0]).astype(np.int)
    v = np.round(cx[:, 1]).astype(np.int)
    d_sensor = -np.ones(len(u))
    d_sensor[visible] = depth_im[v[visible], u[visible]]
    visible = np.logical_and(visible, np.abs(d_sensor-cX[:, 2]) < depth_thresh)

    return cx, visible