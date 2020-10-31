# Copyright (c) Facebook, Inc. and its affiliates.
# Code by Samarth Brahmbhatt

import os
import shutil
import sys

osp = os.path

def remove(p_num):
  for ins in ('use', 'handoff'):
    p_id = 'full{:s}_{:s}'.format(p_num, ins)
    sess_dir = osp.join('..', '..', 'data', 'contactpose_data', p_id)
    for object_name in next(os.walk(sess_dir))[1]:
      obj_dir = osp.join(sess_dir, object_name)
      for filename in next(os.walk(obj_dir))[-1]:
        if '.zip' not in filename:
          continue
        filename = osp.join(obj_dir, filename)
        os.remove(filename)
        print(filename)
      obj_dir = osp.join(obj_dir, 'images_full')
      # if osp.isdir(obj_dir):
      #   shutil.rmtree(obj_dir)
      #   print(obj_dir)
      for camera_name in ('kinect2_left', 'kinect2_right', 'kinect2_middle'):
        cam_dir = osp.join(obj_dir, camera_name)
        filename = osp.join(cam_dir, 'color.mp4')
        if osp.isfile(filename):
          os.remove(filename)
          print(filename)
    for filename in next(os.walk(sess_dir))[-1]:
      filename = osp.join(sess_dir, filename)
      os.remove(filename)
      print(filename)


if __name__ == '__main__':
  remove(sys.argv[1])
