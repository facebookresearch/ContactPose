# Copyright (c) Facebook, Inc. and its affiliates.
# Code by Samarth Brahmbhatt
"""
Calculates and shows the contact probability for hand points
Figure 5(a) in the paper
"""
import os
import matplotlib.pyplot as plt
import numpy as np

import init_paths
from utilities.import_open3d import *
from utilities.dataset import ContactPose, get_object_names
import utilities.misc as mutils

osp = os.path


def calc_hand_contact_prob(p_nums, intents, object_names, contact_thresh=0.4,
                           search_r=15e-3, hand_idx=1):
  """
  hand_idx: 0 for left, 1 for right
  """
  contact_probs = []
  for p_num in p_nums:
    for intent in intents:
      if object_names is None:
        object_names = get_object_names(p_num, intent)
      for object_name in object_names:
        print('{:d} : {:s} : {:s}'.format(p_num, intent, object_name))
        cp = ContactPose(p_num, intent, object_name)
        object_mesh = o3dio.read_triangle_mesh(cp.contactmap_filename)
        v = np.array(object_mesh.vertices)
        c = np.array(object_mesh.vertex_colors)[:, 0]
        c = mutils.texture_proc(c)
        idx = c >= contact_thresh
        v = v[idx]
        
        # read mano
        hand = cp.mano_meshes()[hand_idx]
        if hand is None:
          continue

        h_pc = o3dg.PointCloud()
        h_pc.points = o3du.Vector3dVector(hand['vertices'])
        tree = o3dg.KDTreeFlann(h_pc)

        contact_prob = np.zeros(len(hand['vertices']))
        for vv in v:
          k, idx, dist2 = tree.search_hybrid_vector_3d(vv, search_r, 10)
          for i in range(k):
            # contact_prob[idx[i]] += (1.0/np.sqrt(dist2[i]))
            contact_prob[idx[i]] = 1
        contact_probs.append(contact_prob)
  contact_probs = np.mean(contact_probs, axis=0)
  return contact_probs


def show_hand_contact_prob(contact_prob, hand_idx=1):
  # dummy params
  mp = {
    'pose': np.zeros(15+3),
    'betas': np.zeros(10),
    'hTm': np.eye(4)
  }
  hand = mutils.load_mano_meshes([mp, mp], ContactPose._mano_dicts,
                          flat_hand_mean=True)[hand_idx]
  contact_prob -= contact_prob.min()
  contact_prob /= contact_prob.max()

  contact_prob = plt.cm.bwr(contact_prob)[:, :3]
  h = o3dg.TriangleMesh()
  h.vertices = o3du.Vector3dVector(hand['vertices'])
  h.triangles = o3du.Vector3iVector(hand['faces'])
  h.vertex_colors = o3du.Vector3dVector(contact_prob)
  h.compute_vertex_normals()
  o3dv.draw_geometries([h])


if __name__ == '__main__':
  parser = mutils.default_multiargparse()
  args = parser.parse_args()
  p_nums, intents, object_names, args = mutils.parse_multiargs(args)
  hand_idx = 1
  p = calc_hand_contact_prob(p_nums, intents, object_names, hand_idx=hand_idx)
  show_hand_contact_prob(p, hand_idx=hand_idx)