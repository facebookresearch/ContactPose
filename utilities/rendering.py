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
    mesh_scale: scale factor applied to the mesh
    """
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

  
  def render(self, cTo):
    """
    returns depth map produced by rendering the mesh
    """
    oTc = np.linalg.inv(cTo)
    oTopengl = oTc @ self.cTopengl
    self.scene.set_pose(self.camera_node, oTopengl)
    # TODO: figure out DEPTH_ONLY rendering mode with OSMesa backend
    # DEPTH_ONLY + OSMesa does not work currently
    # so we have to render color also :(
    _, depth = self.renderer.render(self.scene)
    return self.flip_fn(depth)