# Copyright (c) Facebook, Inc. and its affiliates.
# Code by Samarth Brahmbhatt
import os
from scripts.download_data import ContactPoseDownloader

osp = os.path


def startup(data_dir=None):
  default_dir = osp.join('data', 'contactpose_data')
  if data_dir is None:
    data_dir = default_dir
    print('ContactPose data directory is {:s}'.format(data_dir))
    if not osp.isdir(data_dir):
      os.mkdir(data_dir)
  else:
    data_dir = osp.expanduser(data_dir)
    print('ContactPose data directory is {:s}'.format(data_dir))
    if osp.islink(default_dir):
      os.remove(default_dir)
      os.symlink(data_dir, default_dir)
    elif osp.isdir(default_dir) or osp.isfile(default_dir):
      print('{:s} needs to be symlinked to {:s}, please delete it and re-run'.
            format(default_dir, data_dir))
      return
    else:
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
  downloader.download_images(28, 'use', include_objects=['utah_teapot',])


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', default=None,
                      help='Base data dir for the ContactPose dataset')
  args = parser.parse_args()
  startup(args.data_dir)
