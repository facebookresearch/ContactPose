# Copyright (c) Facebook, Inc. and its affiliates.
# Code by Samarth Brahmbhatt
import numpy as np
import logging
import math
import transforms3d.euler as txe
import transforms3d.quaternions as txq
import argparse
import cv2
import matplotlib.pyplot as plt

try:
  from thirdparty.mano.webuser.smpl_handpca_wrapper_HAND_only \
    import load_model as load_mano_model
  MANO_PRESENT = True
except ImportError:
  load_mano_model = None
  MANO_PRESENT = False

if MANO_PRESENT:
  # hacks needed for MANO Python2 code
  import os.path as osp
  import _pickle as cPickle
  import sys
  sys.modules['cPickle'] = cPickle
  sys.path.append(osp.join('thirdparty', 'mano'))
  sys.path.append(osp.join('thirdparty', 'mano', 'webuser'))


def texture_proc(colors, a=0.05, invert=False):
  idx = colors > 0
  ci = colors[idx]
  if len(ci) == 0:
      return colors
  if invert:
      ci = 1 - ci
  # fit a sigmoid
  x1 = min(ci); y1 = a
  x2 = max(ci); y2 = 1-a
  lna = np.log((1 - y1) / y1)
  lnb = np.log((1 - y2) / y2)
  k = (lnb - lna) / (x1 - x2)
  mu = (x2*lna - x1*lnb) / (lna - lnb)
  # apply the sigmoid
  ci = np.exp(k * (ci-mu)) / (1 + np.exp(k * (ci-mu)))
  colors[idx] = ci
  return colors

class MovingAverage:
  def __init__(self):
    self.count = 0
    self.val = 0

  def append(self, v):
    self.val = self.val*self.count + v
    self.count += 1
    self.val /= self.count


def linesegment_from_points(p1, p2):
  n = p2 - p1
  return np.hstack((p1, n))


def get_hand_line_ids():
  line_ids = []
  for finger in range(5):
    base = 4*finger + 1
    line_ids.append([0, base])
    for j in range(3):
        line_ids.append([base+j, base+j+1])
  line_ids = np.asarray(line_ids, dtype=int)
  return line_ids


def rotmat_from_vecs(v1, v2=np.asarray([0, 0, 1])):
  """ 
  Returns a rotation matrix R_1_2
  :param v1: vector in frame 1
  :param v2: vector in frame 2
  :return:
  """
  v1 = v1 / np.linalg.norm(v1)
  v2 = v2 / np.linalg.norm(v2)
  v = np.cross(v2, v1) 
  vx = np.asarray([
    [0,    -v[2], +v[1], 0], 
    [+v[2], 0,    -v[0], 0], 
    [-v[1], +v[0], 0,    0], 
    [0,     0,     0,    0]])
  dotp = np.dot(v1, v2) 

  if np.abs(dotp + 1) < 1e-3:
    R = np.eye(4)
    x = np.cross(v2, [1, 0, 0]) 
    R[:3, :3] = txe.axangle2mat(x, np.pi)
  else:
    R = np.eye(4) + vx + np.dot(vx, vx)/(1+dotp)
  return R


def p_dist_linesegment(p, ls):
  """
  Distance from point p to line segment ls
  p: Nx3
  ls: Mx6 (2 3-dim endpoints of M line segments)
  """
  # NxMx3
  ap = p[:, np.newaxis, :] - ls[np.newaxis, :, :3]
  # 1xMx3
  u = ls[np.newaxis, :, 3:]
  # 1xMx3
  u_norm = u / np.linalg.norm(u, axis=2, keepdims=True)

  # NxM
  proj = np.sum(ap * u_norm, axis=2)

  # point to line distance
  # NxM
  d_line = np.linalg.norm(np.cross(ap, u_norm, axis=2), axis=2)

  # point to endpoint distance
  # NxM
  d_a = np.linalg.norm(ap, axis=2)
  d_b = np.linalg.norm(ap-u, axis=2)
  d_endp = np.minimum(d_a, d_b)

  within_ls = (proj > 0) * (proj < np.linalg.norm(u, axis=2)) * (d_endp < 0.03)
  d_ls = within_ls*d_line + (1-within_ls)*d_endp
  return d_ls


def closest_linesegment_point(l0, l1, p):
  """
  For each point in p, finds the closest point on the list of line segments
  whose endpoints are l0 and l1
  p: N x 3
  l0, l1: M x 3
  out: N x M x 3
  """
  p  = np.broadcast_to(p[:, np.newaxis, :],  (len(p), len(l0), 3))
  l0 = np.broadcast_to(l0[np.newaxis, :, :], (len(p), len(l0), 3))
  l1 = np.broadcast_to(l1[np.newaxis, :, :], (len(p), len(l1), 3))
  
  llen = np.linalg.norm(l1 - l0, axis=-1, keepdims=True)
  lu = (l1 - l0) / llen

  v = p - l0
  d = np.sum(v * lu, axis=-1, keepdims=True)
  d = np.clip(d, a_min=0, a_max=llen)

  out = l0 + d*lu
  return out


def pose_matrix(pose):
  T = np.eye(4)
  T[:3, 3]  = pose['translation']
  T[:3, :3] = txq.quat2mat(pose['rotation'])
  return T


def tform_points(T, X):
  """
  X: Nx3
  T: 4x4 homogeneous
  """
  X = np.vstack((X.T, np.ones(len(X))))
  X = T @ X
  X = X[:3].T
  return X


def project(P, X):
  """
  X: Nx3
  P: 3x4 projection matrix, ContactPose.P or K @ cTo
  returns Nx2 perspective projections
  """
  X = np.vstack((X.T, np.ones(len(X))))
  x = P @ X
  x = x[:2] / x[2]
  return x.T


def get_A(camera_name, W=960, H=540):
  """
  Get the affine transformation matrix applied after 3D->2D projection
  """
  def flipud(H):
      return np.asarray([[1, 0, 0], [0, -1, H], [0, 0, 1]])
  def fliplr(W):
      return np.asarray([[-1, 0, W], [0, 1, 0], [0, 0, 1]])
  def transpose():
      return np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

  if camera_name == 'kinect2_left':
      return np.dot(fliplr(H), transpose())
  elif camera_name == 'kinect2_right':
      return np.dot(flipud(W), transpose())
  elif camera_name == 'kinect2_middle':
      return np.dot(fliplr(W), flipud(H))
  else:
      raise NotImplementedError


def setup_logging(filename=None):
  logging.basicConfig(level=logging.DEBUG)
  root = logging.getLogger()
  if filename is not None:
    root.addHandler(logging.FileHandler(filename, 'w'))
    root.info('Logging to {:s}'.format(filename))


def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True): 
  EPS = np.finfo(float).eps * 4.0
  q0 = np.asarray(quat0) / np.linalg.norm(quat0)
  q1 = np.asarray(quat1) / np.linalg.norm(quat1)
  if fraction == 0.0: 
    return q0 
  elif fraction == 1.0: 
    return q1 
  d = np.dot(q0, q1) 
  if abs(abs(d) - 1.0) < EPS: 
    return q0 
  if shortestpath and d < 0.0: 
    # invert rotation 
    d = -d 
    q1 *= -1.0 
  angle = math.acos(d) + spin * math.pi 
  if abs(angle) < EPS: 
    return q0 
  isin = 1.0 / math.sin(angle) 
  q0 *= math.sin((1.0 - fraction) * angle) * isin 
  q1 *= math.sin(fraction * angle) * isin 
  q0 += q1 
  return q0 


def average_quaternions(qs, ws=None):
  """
  From https://qr.ae/TcwOci
  """
  if ws is None:
    ws = np.ones(len(qs)) / len(qs)
  else:
    assert sum(ws) == 1

  for idx in range(1, len(qs)):
    if np.dot(qs[0], qs[idx]) < 0:
        qs[idx] *= -1

  for i in range(1, len(qs)):
    frac = ws[i] / (ws[i-1] + ws[i]) # weight of qs[i]
    qs[i] = quaternion_slerp(qs[i-1], qs[i], fraction=frac)
    ws[i] = 1 - sum(ws[i+1:])

  return qs[-1]


def default_argparse(require_p_num=True, require_intent=True,
                     require_object_name=True):
  parser = argparse.ArgumentParser()
  parser.add_argument('--p_num', type=int, help='Participant number (1-50)',
                      required=require_p_num)
  parser.add_argument('--intent', choices=('use', 'handoff'),
                      help='Grasp intent', required=require_intent)
  parser.add_argument('--object_name', help="Name of object",
                      required=require_object_name)
  return parser


def default_multiargparse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--p_num',
                      help='Participant numbers, comma or - separated.'
                      'Skipping means all participants',
                      default=None)
  parser.add_argument('--intent', choices=('use', 'handoff', 'use,handoff'),
                      help='Grasp intents, comma separated', default='use,handoff')
  parser.add_argument('--object_name',
                      help="Object names, comma separated, ignore for all objects",
                      default=None)
  return parser


def parse_multiargs(args):
  """
  parses the p_num, intent, and object_name arguments from a parser created
  with default_multiargparse
  """
  from utilities.dataset import get_p_nums
  
  p_nums = args.p_num
  if p_nums is None:
    p_nums = list(range(1, 51))
  elif '-' in p_nums:
    first, last = p_nums.split('-')
    p_nums = list(range(int(first), int(last)+1))
  else:
    p_nums = [int(p) for p in p_nums.split(',')]

  intents = args.intent.split(',')

  object_names = args.object_name
  if object_names is not None:
    object_names = object_names.split(',')
    all_p_nums = []
    for intent in intents:
      for object_name in object_names:
        all_p_nums.extend([pn for pn in p_nums if pn in
                           get_p_nums(object_name, intent)])
    p_nums = list(set(all_p_nums))

  delattr(args, 'p_num')
  delattr(args, 'intent')
  delattr(args, 'object_name')

  return p_nums, intents, object_names, args


def colorcode_depth_image(im):
  assert(im.ndim == 2)
  im = im.astype(float)
  im /= im.max()
  j, i = np.nonzero(im)
  c = im[j, i]
  im = np.zeros((im.shape[0], im.shape[1], 3))
  im[j, i, :] = plt.cm.viridis(c)[:, :3]
  im = (im * 255.0).astype(np.uint8)
  return im


def draw_hands(im, joints, colors=((0, 255, 0), (0, 0, 255)), circle_radius=3,
               line_thickness=2, offset=np.zeros(2, dtype=np.int)):
  if im is None:
    print('Invalid image')
    return im
  if im.ndim == 2:  # depth image
    im = colorcode_depth_image(im)
  for hand_idx, (js, c) in enumerate(zip(joints, colors)):
    if js is None:
      continue
    else:
      js = np.round(js-offset[np.newaxis, :]).astype(np.int)
    for j in js:
      im = cv2.circle(im, tuple(j), circle_radius, c, -1, cv2.LINE_AA)
    for finger in range(5):
      base = 4*finger + 1
      im = cv2.line(im, tuple(js[0]), tuple(js[base]), (0, 0, 0),
                    line_thickness, cv2.LINE_AA)
      for j in range(3):
        im = cv2.line(im, tuple(js[base+j]), tuple(js[base+j+1]),
                      (0, 0, 0), line_thickness, cv2.LINE_AA)
  return im


def draw_object_markers(im, ms, color=(0, 255, 255), circle_radius=3,
                        offset=np.zeros(2, dtype=np.int)):
  if im.ndim == 2:  # depth image
    im = colorcode_depth_image(im)
  for m in np.round(ms).astype(np.int):
    im = cv2.circle(im, tuple(m-offset), circle_radius, color, -1, cv2.LINE_AA)
  return im


def crop_image(im, joints, crop_size, fillvalue=[0]):
  """
  joints: list of 21x2 2D joint locations per each hand
  crops the im into a crop_size square centered at the mean of all joint
  locations
  returns cropped image and top-left pixel position of the crop in the full image
  """
  if im.ndim < 3:
    im = im[:, :, np.newaxis]
  if isinstance(fillvalue, list) or isinstance(fillvalue, np.ndarray):
    fillvalue = np.asarray(fillvalue).astype(im.dtype)
  else:
    fillvalue = np.asarray([fillvalue for _ in im.shape[2]]).astype(im.dtype)

  joints = np.vstack([j for j in joints if j is not None])
  bbcenter = np.round(np.mean(joints, axis=0)).astype(np.int)
  im_crop = np.zeros((crop_size, crop_size, im.shape[2]), dtype=im.dtype)
  tl = bbcenter - crop_size//2
  br = bbcenter + crop_size//2
  tl_crop = np.asarray([0, 0], dtype=np.int)
  br_crop = np.asarray([crop_size, crop_size], dtype=np.int)
  tl_spill = np.minimum(0, tl)
  tl -= tl_spill
  tl_crop -= tl_spill
  br_spill = np.maximum(0, br-np.array([im.shape[1], im.shape[0]]))
  br -= br_spill
  br_crop -= br_spill
  im_crop[tl_crop[1]:br_crop[1], tl_crop[0]:br_crop[0], :] = \
    im[tl[1]:br[1], tl[0]:br[0], :]
  return im_crop.squeeze(), tl


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


def load_mano_meshes(params, model_dicts, oTh=(np.eye(4), np.eye(4)),
                     flat_hand_mean=False):
  if not MANO_PRESENT or model_dicts is None:
    return (None, None)
  
  out = []
  for hand_idx, mp in enumerate(params):
    if mp is None:
      out.append(None)
      continue

    ncomps = len(mp['pose']) - 3
    m = load_mano_model(model_dicts[hand_idx], ncomps=ncomps,
                        flat_hand_mean=flat_hand_mean)
    m.betas[:] = mp['betas']
    m.pose[:]  = mp['pose']

    oTm = oTh[hand_idx] @ mp['hTm']

    vertices = np.array(m)
    vertices = tform_points(oTm, vertices)

    joints = mano2openpose(mano_joints_with_fingertips(m))
    joints = tform_points(oTm, joints)

    out.append({
      'vertices': vertices,
      'joints': joints,
      'faces': np.asarray(m.f),
    })
  return out


def grabcut_mask(src, mask, n_iters=10):
  """
  Refines noisy mask edges using Grabcut on image src
  """
  assert(src.shape[:2] == mask.shape[:2])
  y, x = np.where(mask)
  gmask = np.zeros((src.shape[0], src.shape[1]), dtype=np.uint8)  # GC_BGD
  gmask[y.min():y.max()+1, x.min():x.max()+1] = 2  # GC_PR_BGD
  gmask[y, x] = 3  # GC_PR_FGD

  bgdModel = np.zeros((1,65),np.float64)
  fgdModel = np.zeros((1,65),np.float64)

  gmask, bgdModel, fgdModel = \
      cv2.grabCut(src, gmask, (0, 0, 0, 0), bgdModel, fgdModel, n_iters,
                  mode=cv2.GC_INIT_WITH_MASK)

  mask = np.logical_or(gmask==1, gmask==3)
  return mask