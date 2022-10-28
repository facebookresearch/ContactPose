# Copyright (c) Facebook, Inc. and its affiliates.
# Code by Samarth Brahmbhatt
"""
Preprocesses images for ML training by cropping (RGB and depth), and 
randomizing background (RGB only)
NOTE: Requites rendering setup, see docs/rendering.py
"""
import init_paths
from utilities.dataset import ContactPose, get_object_names
from utilities.rendering import DepthRenderer
import utilities.misc as mutils

import numpy as np
import cv2
import os
from tqdm import tqdm

osp = os.path


def inspect_dir(dirname):
  assert(osp.isdir(dirname))
  print('Inspecting {:s}...'.format(dirname))
  filenames = next(os.walk(dirname))[-1]
  filenames = [osp.join(dirname, f) for f in filenames]
  print('Found {:d} images'.format(len(filenames)))
  return filenames


def preprocess(p_num, intent, object_name, rim_filenames_or_dir, crop_size,
               do_rgb=True, do_depth=True, do_grabcut=True,
               depth_percentile_thresh=30, mask_dilation=5):
  if isinstance(rim_filenames_or_dir, list):
    rim_filenames = rim_filenames_or_dir[:]
  else:
    rim_filenames = inspect_dir(rim_filenames_or_dir)
  cp = ContactPose(p_num, intent, object_name, load_mano=False)

  for camera_name in cp.valid_cameras:
    K = cp.K(camera_name)
    renderer = DepthRenderer(object_name, K, camera_name, 1e-3)
    output_dir = osp.join(cp.data_dir, 'images', camera_name)
    for d in ('color', 'depth', 'projections'):
      dd = osp.join(output_dir, d)
      if not osp.isdir(dd):
        os.makedirs(dd)
    A = mutils.get_A(camera_name)
    print('{:d}:{:s}:{:s}:{:s}'.format(p_num, intent, object_name, camera_name))
    print('Writing to {:s}'.format(output_dir))
    for frame_idx in tqdm(range(len(cp))):
      # read images
      filename = cp.image_filenames('color', frame_idx)[camera_name]
      rgb_im = cv2.imread(filename)
      if rgb_im is None:
        print('Could not read {:s}, skipping frame'.format(filename))
        continue
      filename = cp.image_filenames('depth', frame_idx)[camera_name]
      _, out_filename = osp.split(filename)
      depth_im = cv2.imread(filename, -1)
      if depth_im is None:
        print('Could not read {:s}, skipping frame'.format(filename))
        continue

      # crop images
      joints = cp.projected_hand_joints(camera_name, frame_idx)
      rgb_im, _ = mutils.crop_image(rgb_im, joints, crop_size)
      depth_im, crop_tl = mutils.crop_image(depth_im, joints, crop_size)
      this_A = np.copy(A)
      A = np.asarray([[1, 0, -crop_tl[0]], [0, 1, -crop_tl[1]], [0, 0, 1]]) @ A
      cTo = cp.object_pose(camera_name, frame_idx)
      P = this_A @ K @ cTo[:3]
      
      if do_depth: # save preprocessed depth image
        filename = osp.join(output_dir, 'depth', out_filename)
        cv2.imwrite(filename, depth_im)
      # save projection matrix
      filename = osp.join(output_dir, 'projections',
                          out_filename.replace('.png', '_P.txt'))
      np.savetxt(filename, P)

      # foreground mask
      cxx, visible = renderer.object_visibility_and_projections(cTo)
      cxx -= crop_tl
      cx = np.round(cxx).astype(np.int)
      visible = np.logical_and(visible, cx[:, 0]>=0)
      visible = np.logical_and(visible, cx[:, 1]>=0)
      visible = np.logical_and(visible, cx[:, 0] < rgb_im.shape[1])
      visible = np.logical_and(visible, cx[:, 1] < rgb_im.shape[0])
      cx = cx[visible]

      # save projection information
      filename = osp.join(output_dir, 'projections',
                          out_filename.replace('.png', '_verts.npy'))
      idx = np.where(visible)[0]
      projs = np.vstack((cxx[idx].T, idx)).T
      np.save(filename, projs)
    
      if not do_rgb:
        continue
      
      obj_depths = depth_im[cx[:, 1], cx[:, 0]]
      obj_depths = obj_depths[obj_depths > 0]
      all_depths = depth_im[depth_im > 0]
      if (len(obj_depths) > 0) and (len(all_depths) > 0):
        mthresh = np.median(obj_depths) + 150.0
        pthresh = np.percentile(depth_im[depth_im>0], depth_percentile_thresh)
      else:
        print('Depth image {:s} all 0s, skipping frame'.format(filename))
        continue
      thresh = min(pthresh, mthresh)

      # mask derived from depth
      dmask = 255 * np.logical_and(depth_im > 0, depth_im <= thresh)
      dmask = cv2.dilate(dmask.astype(np.uint8), np.ones(
          (mask_dilation, mask_dilation), dtype=np.uint8))
      # mask derived from color
      cmask_green = np.logical_and(rgb_im[:, :, 1] > rgb_im[:, :, 0],
                                    rgb_im[:, :, 1] > rgb_im[:, :, 2])
      cmask_white = np.mean(rgb_im, axis=2) > 225
      cmask = np.logical_not(np.logical_or(cmask_green, cmask_white))
      mask = np.logical_and(dmask>0, cmask)
      if do_grabcut:
        mask = mutils.grabcut_mask(rgb_im, mask)

      # randomize background
      count = 0
      while count < len(rim_filenames):
        random_idx = np.random.choice(len(rim_filenames))
        random_im = cv2.imread(rim_filenames[random_idx], cv2.IMREAD_COLOR)
        if np.any(np.asarray(random_im.shape[:2]) <= np.asarray(rgb_im.shape[:2])):
          count += 1
          continue
        x = np.random.choice(random_im.shape[1] - rgb_im.shape[1])
        y = np.random.choice(random_im.shape[0] - rgb_im.shape[0])
        random_im = random_im[y:y+rgb_im.shape[0], x:x+rgb_im.shape[1], :]
        break
      else:
        print('ERROR: All random images are smaller than {:d}x{:d}!'.
              format(crop_size, crop_size))
        break
      
      mask = mask[:, :, np.newaxis]
      im = mask*rgb_im + (1-mask)*random_im
      filename = osp.join(output_dir, 'color', out_filename)
      cv2.imwrite(filename, im)
      

def preprocess_all(p_nums, intents, object_names, background_images_dir, *args,
                   **kwargs):
  rim_filenames = inspect_dir(background_images_dir)
  for p_num in p_nums:
    for intent in intents:
      if object_names is None:
        object_names = get_object_names(p_num, intent)
      for object_name in object_names:
        preprocess(p_num, intent, object_name, rim_filenames_or_dir=rim_filenames,
                   *args, **kwargs)


if __name__ == '__main__':
  parser = mutils.default_multiargparse()
  parser.add_argument('--no_rgb', action='store_false', dest='do_rgb')
  parser.add_argument('--no_depth', action='store_false', dest='do_depth')
  parser.add_argument('--background_images_dir', required=True,
                      help='Directory containing background images e.g. COCO')
  parser.add_argument('--crop_size', default=256, type=int)
  parser.add_argument('--no_mask_refinement', action='store_false',
                      dest='do_grabcut',
                      help='No refinement of masks with GrabCut')
  args = parser.parse_args()

  p_nums, intents, object_names, args = mutils.parse_multiargs(args)
  preprocess_all(p_nums, intents, object_names, **vars(args))
