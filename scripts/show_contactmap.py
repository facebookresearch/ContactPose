# Copyright (c) Facebook, Inc. and its affiliates.
# Code by Samarth Brahmbhatt
from open3d import io as o3dio
from open3d import visualization as o3dv
from open3d import utility as o3du
from open3d import geometry as o3dg
import matplotlib.pyplot as plt
import numpy as np

import init_paths
from utilities.dataset import ContactPose
import utilities.misc as mutils


def apply_colormap_to_mesh(mesh, sigmoid_a=0.05, invert=False):
  colors = np.asarray(mesh.vertex_colors)[:, 0]
  colors = mutils.texture_proc(colors, a=sigmoid_a, invert=invert)
  colors = plt.cm.inferno(colors)[:, :3]
  mesh.vertex_colors = o3du.Vector3dVector(colors)
  return mesh


def apply_semantic_colormap_to_mesh(mesh, semantic_idx, sigmoid_a=0.05,
                                    invert=False):
  colors = np.asarray(mesh.vertex_colors)[:, 0]
  colors = mutils.texture_proc(colors, a=sigmoid_a, invert=invert)

  # apply different colormaps based on finger
  mesh_colors = np.zeros((len(colors), 3))
  cmaps = ['Greys', 'Purples', 'Oranges', 'Greens', 'Blues', 'Reds']
  cmaps = [plt.cm.get_cmap(c) for c in cmaps]
  for semantic_id in np.unique(semantic_idx):
    if (len(cmaps) <= semantic_id):
      print('Not enough colormaps, ignoring semantic id {:d}'.format(
          semantic_id))
      continue
    idx = semantic_idx == semantic_id
    mesh_colors[idx] = cmaps[semantic_id](colors[idx])[:, :3]
  mesh.vertex_colors = o3du.Vector3dVector(mesh_colors)
  return mesh


def show_contactmap(p_num, intent, object_name, mode='simple',
                    joint_sphere_radius_mm=4.0, bone_cylinder_radius_mm=2.5,
                    bone_color=np.asarray([224.0, 172.0, 105.0])/255):
  """
  mode =
  simple: just contact map
  simple_hands: skeleton + contact map
  semantic_hands_fingers: skeleton + contact map colored by finger proximity
  semantic_hands_phalanges: skeleton + contact map colored by phalange proximity
  """
  cp = ContactPose(p_num, intent, object_name)

  # read contactmap
  mesh = o3dio.read_triangle_mesh(cp.contactmap_filename)
  mesh.compute_vertex_normals()

  geoms = []
  # apply simple colormap to the mesh
  if 'simple' in mode:
    mesh = apply_colormap_to_mesh(mesh)
    geoms.append(mesh)

  if 'hands' in mode:
    # read hands
    line_ids = mutils.get_hand_line_ids()
    joint_locs = cp.hand_joints()
    
    # show hands
    hand_colors = [[0, 1, 0], [1, 0, 0]]
    for hand_idx, hand_joints in enumerate(joint_locs):
      if hand_joints is None:
        continue

      # joint locations
      for j in hand_joints:
        m = o3dg.TriangleMesh.create_sphere(radius=joint_sphere_radius_mm*1e-3,
                                            resolution=10)
        T = np.eye(4)
        T[:3, 3] = j
        m.transform(T)
        m.paint_uniform_color(hand_colors[hand_idx])
        m.compute_vertex_normals()
        geoms.append(m)

      # connecting lines
      for line_idx, (idx0, idx1) in enumerate(line_ids):
        bone = hand_joints[idx0] - hand_joints[idx1]
        h = np.linalg.norm(bone)
        l = o3dg.TriangleMesh.create_cylinder(radius=bone_cylinder_radius_mm*1e-3,
                                      height=h, resolution=10)
        T = np.eye(4)
        T[2, 3] = -h/2.0
        l.transform(T)
        T = mutils.rotmat_from_vecs(bone, [0, 0, 1]) 
        T[:3, 3] = hand_joints[idx0]
        l.transform(T)
        l.paint_uniform_color(bone_color)
        l.compute_vertex_normals()
        geoms.append(l)

    if 'semantic' in mode:
      n_lines_per_hand = len(line_ids)
      n_parts_per_finger = 4
      # find line equations for hand parts
      lines = []
      for hand_joints in joint_locs:
        if hand_joints is None:
          continue
        for line_id in line_ids:
          a = hand_joints[line_id[0]]
          n = hand_joints[line_id[1]] - hand_joints[line_id[0]]
          n /= np.linalg.norm(n)
          lines.append(np.hstack((a, n)))
      lines = np.asarray(lines)

      ops = np.asarray(mesh.vertices)
      d_lines = mutils.p_dist_linesegment(ops, lines)
      line_idx = np.argmin(d_lines, axis=1) % n_lines_per_hand
      finger_idx, part_idx = divmod(line_idx, n_parts_per_finger)
      if 'phalanges' in mode:
          mesh = apply_semantic_colormap_to_mesh(mesh, part_idx)
      elif 'fingers' in mode:
          mesh = apply_semantic_colormap_to_mesh(mesh, finger_idx)
      geoms.append(mesh)
  elif 'mano' in mode:
    for hand in cp.mano_meshes():
      if hand is None:
        continue
      mesh = o3dg.TriangleMesh()
      mesh.vertices = o3du.Vector3dVector(hand['vertices'])
      mesh.triangles = o3du.Vector3iVector(hand['faces'])
      mesh.paint_uniform_color(bone_color)
      mesh.compute_vertex_normals()
      geoms.append(mesh)
  
  o3dv.draw_geometries(geoms)


if __name__ == '__main__':
  import sys
  parser = mutils.default_argparse()
  parser.add_argument('--mode', help='Contact Map mode', default='simple_hands',
    choices=('simple', 'simple_mano', 'simple_hands', 'semantic_hands_fingers',
             'semantic_hands_phalanges'))
  args = parser.parse_args()
  if args.object_name == 'hands':
    print('hands do not have a contact map')
    sys.exit(0)
  elif args.object_name == 'palm_print':
    print('Forcing mode to simple since palm_print does not have hand pose')
    args.mode = 'simple'
  show_contactmap(args.p_num, args.intent, args.object_name, args.mode)