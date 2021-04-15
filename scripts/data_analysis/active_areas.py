# Copyright (c) Facebook, Inc. and its affiliates.
# Code by Samarth Brahmbhatt
"""
 Discovers 'active areas' i.e. areas on the object surface most frequently 
 touched by a certain part of the hand. See Figure 7 in the paper
 https://arxiv.org/pdf/2007.09545.pdf.
"""
import init_paths
from utilities.import_open3d import *  # need to import open3d before others
import json
from matplotlib import cm
import numpy as np
import os
from random import shuffle
from utilities.dataset import get_p_nums
import utilities.misc as mutils

osp = os.path


def discover_active_areas(finger_idx, part_idx, object_name, intent, p_nums=None,
                          color_thresh=0.4):
  """
  finger_idx: 0->4 : thumb->little
  part_idx: 0->3 : proximal to distal phalanges, 3 = finger tip
  """
  p_nums = p_nums or get_p_nums(object_name, intent)
  shuffle(p_nums)
  data_dir = osp.join('data', 'contactpose_data')

  # read object mesh
  vertices = None
  for p_num in p_nums:  
    filename = osp.join(data_dir, f'full{p_num}_{intent}', object_name,
                        f'{object_name}.ply')
    if osp.isfile(filename):
      mesh = o3dio.read_triangle_mesh(filename)
    else:
      print('{:s} does not exist'.format(filename))
      continue
    vertices = np.asarray(mesh.vertices)
    break
  if vertices is None:
    print("no object model found")
    return

  line_ids = mutils.get_hand_line_ids()
  n_lines_per_hand = len(line_ids)
  n_parts_per_finger = 4

  touched_by_part = np.zeros(len(vertices))
  count = 0
  for p_num in p_nums:
    print(f'Processing full{p_num}_{intent} {object_name}')
    
    # read contact from the mesh
    filename = osp.join(data_dir, f'full{p_num}_{intent}', object_name,
                        f'{object_name}.ply')
    if osp.isfile(filename):
      mesh = o3dio.read_triangle_mesh(filename)
    else:
      print('{:s} does not exist'.format(filename))
      continue
    tex = np.asarray(mesh.vertex_colors)[:, 0]
    tex = mutils.texture_proc(tex)
    
    # read joints
    filename = osp.join(data_dir, f'full{p_num}_{intent}', object_name,
                        'annotations.json')
    try:
      with open(filename, 'r') as f:
        annotations = json.load(f)
    except FileNotFoundError:
      print('{:s} does not exist'.format(filename))
      continue

    ds = []
    for hand_idx, hand in enumerate(annotations['hands']):
      if hand['valid']:
        joints = np.asarray(hand['joints'])
        l0 = joints[line_ids[:, 0]]
        l1 = joints[line_ids[:, 1]]
        pl = mutils.closest_linesegment_point(l0, l1, vertices)
        d  = pl - vertices[:, np.newaxis, :]
        d  = np.linalg.norm(d, axis=2)
      else:
        d = np.inf * np.ones((len(vertices), n_lines_per_hand))
      ds.append(d)
    ds = np.hstack(ds)

    hand_idxs, line_idxs = divmod(np.argmin(ds, axis=1), n_lines_per_hand)
    finger_idxs, part_idxs = divmod(line_idxs, n_parts_per_finger)

    this_touched_by_part = np.logical_and(
        tex > color_thresh, np.logical_and(hand_idxs >= 0,
          np.logical_and(finger_idxs == finger_idx, part_idxs == part_idx)))
    touched_by_part += this_touched_by_part
    count += 1

  touched_by_part /= count
  touched_by_part /= touched_by_part.max()
  filename = osp.join('data',
    f'{object_name}_{intent}_{finger_idx}_{part_idx}_active_areas.npy')
  np.save(filename, touched_by_part)
  print('{:s} saved'.format(filename))


def show_active_areas(finger_idx, part_idx, object_name, intent):
  filename = osp.join('data', 'object_models', f'{object_name}.ply')
  mesh = o3dio.read_triangle_mesh(filename)
  mesh.compute_vertex_normals()

  filename = osp.join('data',
    f'{object_name}_{intent}_{finger_idx}_{part_idx}_active_areas.npy')
  c = np.load(filename)
  mesh.vertex_colors = o3du.Vector3dVector(cm.bwr(c)[:, :3])
  o3dv.draw_geometries([mesh])


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--finger_idx', type=int, required=True,
                      help='0->4 : thumb->little', choices=(0, 1, 2, 3, 4))
  parser.add_argument('--part_idx', type=int, required=True, choices=(0, 1, 2, 3),
                      help='0->3 : proximal to distal phalanges, 3 = finger tip')
  parser.add_argument('--object_name', required=True)
  parser.add_argument('--intent', required=True, choices=('use', 'handoff'))
  parser.add_argument('--p_nums', default='1-50',
                      help='Participant numbers, comma or - separated.'
                      'Skipping means all participants')
  parser.add_argument('--show', action='store_true')
  args = parser.parse_args()

  p_nums = args.p_nums
  if '-' in p_nums:
    first, last = p_nums.split('-')
    p_nums = list(range(int(first), int(last)+1))
  else:
    p_nums = [int(p) for p in p_nums.split(',')]

  if args.show:
    show_active_areas(args.finger_idx, args.part_idx, args.object_name,
        args.intent)
  else:
    discover_active_areas(args.finger_idx, args.part_idx, args.object_name,
                          args.intent, p_nums)
