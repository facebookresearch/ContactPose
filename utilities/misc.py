import numpy as np
import logging
import math
import transforms3d.euler as txe
import transforms3d.quaternions as txq
import argparse
import cv2
from thirdparty.mano.webuser.smpl_handpca_wrapper_HAND_only \
  import load_model as load_mano_model

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
  ls: Mx6
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


def project(K, X):
  """
  X: Nx3
  K: 3x3
  returns Nx2 perspective projections
  """
  x = K @ X.T
  x = x[:2] / x[2]
  return x.T


def get_A(camera_name, W, H):
  """
  Get the euclidean transform matrix representing the camera orientation
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

  for i in xrange(1, len(qs)):
    frac = ws[i] / (ws[i-1] + ws[i]) # weight of qs[i]
    qs[i] = quaternion_slerp(qs[i-1], qs[i], fraction=frac)
    ws[i] = 1 - sum(ws[i+1:])

  return qs[-1]


def default_argparse(require_p_num=True, require_intent=True,
                     require_object_name=True):
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--p_num', type=int, help='Participant number (1-50)',
                      required=require_p_num)
  parser.add_argument('--intent', choices=('use', 'handoff'),
                      help='Grasp intent', required=require_intent)
  parser.add_argument('--object_name', help="Name of object",
                      required=require_object_name)
  return parser


def draw_hands(im, joints, colors=((0, 255, 0), (0, 0, 255)), circle_radius=3,
               line_thickness=2):
  if im is None:
    print('Invalid image')
    return im
  for hand_idx, (js, c) in enumerate(zip(joints, colors)):
    if js is None:
      continue
    else:
      js = np.round(js).astype(np.int)
    for j in js:
      im = cv2.circle(im, (j[0], j[1]), circle_radius, c, -1, cv2.LINE_AA)
    for finger in range(5):
      base = 4*finger + 1
      im = cv2.line(im, (js[0][0], js[0][1]),
                    (js[base][0], js[base][1]), (0, 0, 0),
                    line_thickness, cv2.LINE_AA)
      for j in range(3):
        im = cv2.line(im,
                      (js[base+j][0], js[base+j][1]),
                      (js[base+j+1][0], js[base+j+1][1]),
                      (0, 0, 0), line_thickness, cv2.LINE_AA)
  return im


def draw_object_markers(im, ms, color=(0, 255, 255), circle_radius=3):
  for m in np.round(ms).astype(np.int):
    im = cv2.circle(im, tuple(m), circle_radius, color, -1, cv2.LINE_AA)
  return im


def openpose2mano(o, n_joints_per_finger=4):
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
