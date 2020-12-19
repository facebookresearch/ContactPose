# Copyright (c) Facebook, Inc. and its affiliates.
# Code by Samarth Brahmbhatt
"""
script to download ContactPose data from Dropbox
URLs in data/urls.json
"""
import init_paths
import cv2
import os
import json
import shutil
from tqdm.autonotebook import tqdm
import utilities.networking as nutils
from zipfile import ZipFile

osp = os.path


def is_nonempty_dir(dir):
  if osp.isdir(dir):
    return next(os.scandir(dir), None) is not None
  else:
    return False


class ContactPoseDownloader(object):
  def __init__(self):
    self.data_dir = osp.join('data', 'contactpose_data')
    if not osp.isdir(self.data_dir):
      os.makedirs(self.data_dir)
      print('Created {:s}'.format(self.data_dir))
    with open(osp.join('data', 'urls.json'), 'r') as f:
      self.urls = json.load(f)


  @staticmethod
  def _unzip_and_del(filename, dst_dir=None, progress=True, filter_fn=None):
    if dst_dir is None:
      dst_dir, _ = osp.split(filename)
    if len(dst_dir) == 0:
      dst_dir = '.'
    with ZipFile(filename) as f:
      # members = None means everything
      members = None if filter_fn is None else \
          list(filter(filter_fn, f.namelist()))
      f.extractall(dst_dir, members=members)
    os.remove(filename)


  def download_grasps(self):
    filename = osp.join(self.data_dir, 'grasps.zip')
    print('Downloading grasps...')
    if not nutils.download_url(self.urls['grasps'], filename):
      print('Download unsuccessful')
      return
    print('Extracting...')
    self._unzip_and_del(filename, self.data_dir)
    p_ids = next(os.walk(self.data_dir))[1]
    for p_id in tqdm(p_ids):
      if 'full' not in p_id:
        continue
      sess_dir = osp.join(self.data_dir, p_id)
      for filename in next(os.walk(sess_dir))[-1]:
        if '.zip' not in filename:
          continue
        self._unzip_and_del(osp.join(sess_dir, filename), progress=False)


  def download_contact_maps(self, p_num, intent):
    p_id = 'full{:d}_{:s}'.format(p_num, intent)
    filename = osp.join(self.data_dir, '{:s}_contact_maps.zip'.format(p_id))
    print('Downloading {:d} {:s} contact maps...'.format(p_num, intent))
    if not nutils.download_url(self.urls['contact_maps'][p_id], filename):
      print('Download unsuccessful')
      return
    print('Extracting...')
    self._unzip_and_del(filename, self.data_dir)

  
  def download_markers(self):
    filename = osp.join('data', 'markers.zip')
    print('Downloading 3D model marker locations...')
    if not nutils.download_url(self.urls['object_marker_locations'], filename):
      print('Download unsuccessful')
      return
    print('Extracting...')
    self._unzip_and_del(filename, osp.join('data', 'object_marker_locations'))

  
  def download_3d_models(self):
    filename = osp.join('data', '3Dmodels.zip')
    print('Downloading 3D models...')
    if not nutils.download_url(self.urls['object_models'], filename):
      print('Download unsuccessful')
      return
    print('Extracting...')
    self._unzip_and_del(filename, osp.join('data', 'object_models'))

  
  def download_depth_images(self, p_num, intent, dload_dir,
                            include_objects=None):
    self.download_images(p_num, intent, dload_dir, include_objects,
                         download_color=False, download_depth=True)

  
  def download_color_images(self, p_num, intent, dload_dir,
                            include_objects=None):
    self.download_images(p_num, intent, dload_dir, include_objects,
                         download_color=True, download_depth=False)
  
  
  def download_images(self, p_num, intent, dload_dir,
                      include_objects=None, download_color=True,
                      download_depth=True):
    assert osp.isdir(dload_dir),\
      'Image download dir {:s} does not exist'.format(dload_dir)
    p_id = 'full{:d}_{:s}'.format(p_num, intent)
    if download_color and (not download_depth):
      urls = self.urls['videos']['color']
    else:
      urls = self.urls['images']
    
    # check if already extracted
    dirs_to_check = []
    if download_color:
      dirs_to_check.append('color')
    if download_depth:
      dirs_to_check.append('depth')
    ok = True
    if osp.isdir(osp.join(self.data_dir, p_id)):
      sess_dir = osp.join(self.data_dir, p_id)
      for object_name in next(os.walk(sess_dir))[1]:
        if include_objects is not None and object_name not in include_objects:
          continue
        images_dir = osp.join(sess_dir, object_name, 'images_full')
        if not osp.isdir(images_dir):
          continue
        for cam_name in next(os.walk(images_dir))[1]:
          for check_name in dirs_to_check:
            check_dir = osp.join(images_dir, cam_name, check_name)
            if is_nonempty_dir(check_dir):
              print('{:s} {:s} already has extracted images, please delete {:s}'.
                    format(p_id, object_name, check_dir))
              ok = False
    if not ok:
      return
    
    # download and extract
    sess_dir = osp.join(dload_dir, p_id)
    if not osp.isdir(sess_dir):
      print('Creating {:s}'.format(sess_dir))
    os.makedirs(sess_dir, exist_ok=True)
    print('Downloading {:s} images...'.format(p_id))
    object_names = list(urls[p_id].keys())
    if include_objects is None:
      include_objects = object_names[:]
    filenames_to_extract = {}
    for object_name in tqdm(include_objects):
      if object_name not in object_names:
        print('{:d} {:s} does not have {:s}'.format(p_num, intent, object_name))
        continue
      filename = osp.join(sess_dir, '{:s}_images.zip'.format(object_name))
      url = urls[p_id][object_name]
      print(object_name)
      if nutils.download_url(url, filename):
        filenames_to_extract[object_name] = filename
      else:
        print('{:s} {:s} Download unsuccessful'.format(p_id, object_name))
        return
    
    print('Extracting...')
    for object_name, filename in tqdm(filenames_to_extract.items()):
      obj_dir = osp.join(sess_dir, object_name)
      os.makedirs(obj_dir, exist_ok=True)
      self._unzip_and_del(filename, obj_dir)
      for filename in next(os.walk(obj_dir))[-1]:
        if download_color and (not download_depth):
          if '.mp4' not in filename:
            continue
          camera_name = filename.replace('.mp4', '')
          video_filename = osp.join(obj_dir, filename)
          im_dir = osp.join(obj_dir, 'images_full', camera_name, 'color')
          os.makedirs(im_dir, exist_ok=True)
          cap = cv2.VideoCapture(video_filename)
          if not cap.isOpened():
            print('Could not read {:s}'.format(video_filename))
            return
          count = 0
          while True:
            ok, im = cap.read()
            if not ok:
              break
            filename = osp.join(im_dir, 'frame{:03d}.png'.format(count))
            cv2.imwrite(filename, im)
            count += 1
          os.remove(video_filename)
        else:
          if '.zip' not in filename:
            continue
          filter_fn = (lambda x: 'color' not in x) if (not download_color) \
              else None
          self._unzip_and_del(osp.join(obj_dir, filename), progress=False,
                              filter_fn=filter_fn)

      # symlink
      if osp.realpath(dload_dir) != osp.realpath(self.data_dir):
        src = osp.join(obj_dir, 'images_full')
        dst_dir = osp.join(self.data_dir, p_id, object_name)
        if not osp.isdir(dst_dir):
          os.makedirs(dst_dir)
        dst = osp.join(dst_dir, 'images_full')
        os.symlink(src, dst)


if __name__ == '__main__':
  import argparse
  import sys
  from itertools import product
  parser = argparse.ArgumentParser()
  parser.add_argument('--type', choices=('grasps', 'markers', '3Dmodels',
                                         'color_images', 'depth_images',
                                         'images', 'contact_maps'),
                      required=True)
  parser.add_argument('--p_nums', default=None,
                      help='Participant numbers E.g. 1, 1,2, or 1-5')
  parser.add_argument('--intents', default='use,handoff',
                      help='use, handoff, or use,handoff')
  parser.add_argument('--object_names', default=None,
                      help='Comma separated object names. Used only for image '+\
                        'download. All other types download data for all '+\
                        'objects in that particular p_num, intent combo')
  parser.add_argument('--images_dload_dir',
                      default=osp.join('data', 'contactpose_data'),
                      help='Directory where images will be downloaded. '
                      'They will be symlinked to the appropriate location')
  args = parser.parse_args()

  downloader = ContactPoseDownloader()
  if args.type == 'grasps':
    downloader.download_grasps()
    sys.exit(0)
  elif args.type == 'markers':
    downloader.download_markers()
    sys.exit(0)
  elif args.type == '3Dmodels':
    downloader.download_3d_models()
    sys.exit(0)

  assert(args.p_nums is not None)
  if '-' in args.p_nums:
    start, finish = args.p_nums.split('-')
    nums = list(range(int(start), int(finish)+1))
  elif ',' in args.p_nums:
    nums = [int(n) for n in args.p_nums.split(',')]
  else:
    nums = [int(args.p_nums)]
  intents = args.intents.split(',')
  include_objects = args.object_names
  if include_objects is not None:
    include_objects = include_objects.split(',')
    include_objects = list(set(include_objects))  # remove duplicates
  
  for p_num, intent in product(nums, intents):
    p_id = 'full{:d}_{:s}'.format(p_num, intent)
    print('####### {:s} #######'.format(p_id))
    if args.type == 'contact_maps':
      downloader.download_contact_maps(p_num, intent)
    elif args.type == 'color_images':
      downloader.download_color_images(p_num, intent,
                                 osp.expanduser(args.images_dload_dir),
                                 include_objects=include_objects)
    elif args.type == 'depth_images':
      downloader.download_depth_images(p_num, intent,
                                 osp.expanduser(args.images_dload_dir),
                                 include_objects=include_objects)
    elif args.type == 'images':
      downloader.download_images(p_num, intent,
                                 osp.expanduser(args.images_dload_dir),
                                 include_objects=include_objects)
    else:
      raise NotImplementedError
