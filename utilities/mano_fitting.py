# Copyright (c) Facebook, Inc. and its affiliates.
# Code by Samarth Brahmbhatt
from utilities.import_open3d import *
from open3d import pipelines
import utilities.misc as mutils
assert(mutils.load_mano_model is not None)

import numpy as np
import chumpy as ch
import os
import json
import transforms3d.quaternions as txq
import pickle

osp = os.path
o3dr = pipelines.registration


def mano_param_dict(n_pose_params, n_betas=10):
  out = {
    'pose': [0.0 for _ in range(n_pose_params+3)],
    'betas': [0.0 for _ in range(n_betas)],
    'valid': False,
    'mTc': {
      'translation': [0.0, 0.0, 0.0],
      'rotation': [1.0, 0.0, 0.0, 0.0],
    }
  }
  return out


def openpose2mano(o, n_joints_per_finger=4):
  """
  convert joints from openpose format to MANO format
  """
  finger_o2m = {0: 4, 1: 0, 2: 1, 3: 3, 4: 2}
  m = np.zeros((5*n_joints_per_finger+1, 3))
  m[0] = o[0]
  for ofidx in range(5):
    for jidx in range(n_joints_per_finger):
      oidx = 1 + ofidx*4 + jidx
      midx = 1 + finger_o2m[ofidx]*n_joints_per_finger + jidx
      m[midx] = o[oidx]
  return np.array(m)


# m2o
# 0->1, 1->2, 2->4, 3->3, 4->0
def mano2openpose(m, n_joints_per_finger=4):
  """
  convert joints from MANO format to openpose format
  """
  finger_o2m = {0: 4, 1: 0, 2: 1, 3: 3, 4: 2}
  finger_m2o = {v: k for k,v in finger_o2m.items()}
  o = np.zeros((5*n_joints_per_finger+1, 3))
  o[0] = m[0]
  for mfidx in range(5):
    for jidx in range(n_joints_per_finger):
      midx = 1 + mfidx*4 + jidx
      oidx = 1 + finger_m2o[mfidx]*n_joints_per_finger + jidx
      o[oidx] = m[midx]
  return o


def get_palm_joints(p, n_joints_per_finger=4):
  """
  get the 6 palm joints (root + base of all 5 fingers)
  """
  idx = [0]
  for fidx in range(5):
    idx.append(1 + fidx*n_joints_per_finger)
  return p[idx]


def mano_joints_with_fingertips(m):
  """
  get joints from MANO model
  MANO model does not come with fingertip joints, so we have selected vertices
  that correspond to fingertips
  """
  fingertip_idxs = [333, 444, 672, 555, 745]
  out = [m.J_transformed[0]]
  for fidx in range(5):
    for jidx in range(4):
      if jidx < 3:
        idx = 1 + fidx*3 + jidx
        out.append(m.J_transformed[idx])
      else:
        out.append(m[fingertip_idxs[fidx]])
  return out


def register_pcs(src, tgt, verbose=True):
  """
  registers two pointclouds by rigid transformation
  target_x = target_T_source * source_x
  """
  assert(len(src) == len(tgt))
  ps = o3dg.PointCloud()
  ps.points = o3du.Vector3dVector(src)
  pt = o3dg.PointCloud()
  pt.points = o3du.Vector3dVector(tgt)
  c = [[i, i] for i in range(len(src))]
  c = o3du.Vector2iVector(c)
  r = o3dr.TransformationEstimationPointToPoint()
  r.with_scaling = False
  if verbose:
    print('Rigid registration RMSE (before) = {:f}'.
          format(r.compute_rmse(ps, pt, c)))
  tTs = r.compute_transformation(ps, pt, c)
  pst = ps.transform(tTs)
  if verbose:
    print('Rigid registration RMSE (after) = {:f}'.
          format(r.compute_rmse(pst, pt, c)))
  return tTs


class MANOFitter(object):
  _mano_dicts = None
  
  def __init__(self):
    if MANOFitter._mano_dicts is None:
      MANOFitter._mano_dicts = []
      for hand_name in ('LEFT', 'RIGHT'):
        filename = osp.join('thirdparty', 'mano', 'models',
                            'MANO_{:s}.pkl'.format(hand_name))
        with open(filename, 'rb') as f:
          MANOFitter._mano_dicts.append(pickle.load(f, encoding='latin1'))

  @staticmethod 
  def fit_joints(both_joints, n_pose_params=15, shape_sigma=10.0,
                save_filename=None):
    """
    Fits the MANO model to hand joint 3D locations
    both_jonts: tuple of length 2, 21 joints per hand, e.g. output of ContactPose.hand_joints()
    n_pose_params: number of pose parameters (excluding 3 global rotation params)
    shape_sigma: reciprocal of shape regularization strength
    save_filename: file where the fitting output will be saved in JSON format
    """
    mano_params = []
    for hand_idx, joints in enumerate(both_joints):
      if joints is None:  # hand is not present
        mano_params.append(mano_param_dict(n_pose_params))  # dummy
        continue
      cp_joints = openpose2mano(joints)

      # MANO model
      m = mutils.load_mano_model(MANOFitter._mano_dicts[hand_idx],
                                 ncomps=n_pose_params, flat_hand_mean=False)
      m.betas[:] = np.zeros(m.betas.size)
      m.pose[:]  = np.zeros(m.pose.size)
      mano_joints = mano_joints_with_fingertips(m)
      mano_joints_np = np.array([[float(mm) for mm in m] for m in mano_joints])

      # align palm
      cp_palm = get_palm_joints(np.asarray(cp_joints))
      mano_palm = get_palm_joints(np.asarray(mano_joints_np))
      mTc = register_pcs(cp_palm, mano_palm)
      cp_joints = np.dot(mTc, np.vstack((cp_joints.T, np.ones(len(cp_joints)))))
      cp_joints = cp_joints[:3].T
      cp_joints = ch.array(cp_joints)

      # set up objective
      objective = [m-c for m,c in zip(mano_joints, cp_joints)]
      mean_betas = ch.array(np.zeros(m.betas.size))
      objective.append((m.betas - mean_betas) / shape_sigma)
      # optimize
      ch.minimize(objective, x0=(m.pose, m.betas, m.trans), method='dogleg')

      p = mano_param_dict(n_pose_params)
      p['pose']  = np.array(m.pose).tolist()
      p['betas'] = np.array(m.betas).tolist()
      p['valid'] = True
      p['mTc']['translation'] = (mTc[:3, 3] - np.array(m.trans)).tolist()
      p['mTc']['rotation'] = txq.mat2quat(mTc[:3, :3]).tolist()
      mano_params.append(p)

      # # to access hand mesh vertices and faces
      # vertices = np.array(m.r)
      # vertices = mutils.tform_points(np.linalg.inv(mTc), vertices)
      # faces = np.array(m.f)

    if save_filename is not None:
      with open(save_filename, 'w') as f:
        json.dump(mano_params, f, indent=4, separators=(',', ':'))
      print('{:s} written'.format(save_filename))
    return mano_params
