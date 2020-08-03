# Copyright (c) Facebook, Inc. and its affiliates.
# Code by Samarth Brahmbhatt
import os
from scripts.download_data import ContactPoseDownloader

osp = os.path


def startup(data_dir=None, default_dir=osp.join('data', 'contactpose_data')):
  # check that the provided data_dir is OK
  if data_dir is not None:
    assert data_dir!=default_dir, \
      "If you provide --data_dir, it must not be {:s}".format(default_dir)
    assert osp.isdir(data_dir), "If you provide --data_dir, it must exist"
  else:
    data_dir = default_dir
    if not osp.isdir(data_dir):
      if osp.isfile(data_dir) or osp.islink(data_dir):
        os.remove(data_dir)
        print('Removed file {:s}'.format(data_dir))
      os.mkdir(data_dir)

  # symlink for easy access
  if data_dir != default_dir:
    if osp.islink(default_dir):
      os.remove(default_dir)
      print('Removed symlink {:s}'.format(default_dir))
    os.symlink(data_dir, default_dir)
    print('Symlinked to {:s} for easy access'.format(default_dir))
  
  downloader = ContactPoseDownloader()

  # download 3D models and marker locations
  downloader.download_3d_models()
  downloader.download_markers()

  # download all 3D joint, object pose, camera calibration data
  downloader.download_grasps()

  # download contact maps for participant 28, 'use' grasps
  downloader.download_contact_maps(28, 'use')

  # download RGB-D images for participant 28, mouse 'use' grasp 
  downloader.download_images(28, 'use', data_dir,
                             include_objects=('utah_teapot',))


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', default=None,
                      help='Base data dir for the ContactPose dataset')
  args = parser.parse_args()
  startup(args.data_dir)
