"""
script to download ContactPose data from Dropbox
URLs in data/urls.json
"""
import os
import json
from tqdm.autonotebook import tqdm
import requests
from zipfile import ZipFile

osp = os.path


class ContactPoseDownloader(object):
  def __init__(self):
    self.data_dir = osp.join('data', 'contactpose_data')
    if not osp.isdir(self.data_dir):
      os.makedirs(self.data_dir)
      print('Created {:s}'.format(self.data_dir))
    with open(osp.join('data', 'urls.json'), 'r') as f:
      self.urls = json.load(f)


  @staticmethod
  def _path_level(path):
    return len(path.strip(osp.sep).split(osp.sep))

  
  @staticmethod
  def _download_url(url, filename, progress=True):
    """
    taken from https://stackoverflow.com/a/37573701
    """
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)
    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    if progress:
      t=tqdm(total=total_size, unit='iB', unit_scale=True)
    done = True
    datalen = 0
    with open(filename, 'wb') as f:
      itr = r.iter_content(block_size)
      while True:
        try:
          try:
            data = next(itr)
          except StopIteration:
            break
          if progress:
            t.update(len(data))
          datalen += len(data)
          f.write(data)
        except KeyboardInterrupt:
          done = False
          print('Cancelled')
    if progress:
      t.close()
    if (not done) or (total_size != 0 and datalen != total_size):
      print("ERROR, something went wrong")
      try:
        os.remove(filename)
      except OSError as e:
        print(e)
      return False
    else:
      return True

  
  @staticmethod
  def _unzip_and_del(filename, dst_dir=None, progress=True):
    if dst_dir is None:
      dst_dir, _ = osp.split(filename)
    if len(dst_dir) == 0:
      dst_dir = '.'
    with ZipFile(filename) as f:
      f.extractall(dst_dir)
    os.remove(filename)


  def download_grasps(self):
    filename = osp.join(self.data_dir, 'grasps.zip')
    print('Downloading grasps...')
    if not self._download_url(self.urls['grasps'], filename):
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
    print('Downloading {:s} contact maps...'.format(p_id))
    if not self._download_url(self.urls['contact_maps'][p_id], filename):
      print('Download unsuccessful')
      return
    print('Extracting...')
    self._unzip_and_del(filename, self.data_dir)

  
  def download_markers(self):
    filename = osp.join('data', 'markers.zip')
    print('Downloading 3D model marker locations...')
    if not self._download_url(self.urls['object_marker_locations'], filename):
      print('Download unsuccessful')
      return
    print('Extracting...')
    self._unzip_and_del(filename, osp.join('data', 'object_marker_locations'))


  def download_images(self, p_num, intent, dload_dir,
                      include_objects=None):
    p_id = 'full{:d}_{:s}'.format(p_num, intent)
    # check if already extracted
    if osp.isdir(osp.join(self.data_dir, p_id)):
      sess_dir = osp.join(self.data_dir, p_id)
      for object_name in next(os.walk(sess_dir))[1]:
        if include_objects is not None and object_name not in include_objects:
          continue
        images_dir = osp.join(sess_dir, object_name, 'images_full')
        if osp.isdir(images_dir):
          print('{:s} {:s} already has extracted images, please delete {:s}'.
                format(p_id, object_name, images_dir))
          return
    
    # download and extract
    sess_dir = osp.join(dload_dir, p_id)
    if not osp.isdir(sess_dir):
      os.makedirs(sess_dir)
      print('Created {:s}'.format(sess_dir))
    print('Downloading {:s} images...'.format(p_id))
    object_names = list(self.urls['images'][p_id].keys())
    if include_objects is not None:
      object_names = [o for o in object_names if o in include_objects]
    filenames = []
    for object_name in tqdm(object_names):
      filename = osp.join(sess_dir, '{:s}_images.zip'.format(object_name))
      url = self.urls['images'][p_id][object_name]
      print(object_name)
      if self._download_url(url, filename):
        filenames.append(filename)
      else:
        print('{:s} {:s} Download unsuccessful'.format(p_id, object_name))
        return
    
    print('Extracting...')
    for object_name, filename in tqdm(zip(object_names, filenames)):
      obj_dir = osp.join(sess_dir, object_name)
      if not osp.isdir(obj_dir):
        os.mkdir(obj_dir)
      self._unzip_and_del(filename, obj_dir)
      for filename in next(os.walk(obj_dir))[-1]:
        if '.zip' not in filename:
          continue
        self._unzip_and_del(osp.join(obj_dir, filename), progress=False)
      # symlink
      if dload_dir != self.data_dir:
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
  parser.add_argument('--type', choices=('grasps', 'images', 'contact_maps'),
                      required=True)
  parser.add_argument('--p_nums', default=None,
                      help='Participant numbers E.g. 1, 1,2, or 1-5')
  parser.add_argument('--intents', default='use,handoff',
                      help='use, handoff, or use,handoff')
  parser.add_argument('--object_names', default=None,
                      help='Comma separated object names. Used only for image '+\
                        'download. All other types download data for all '+\
                        'objects in that particular p_num, intent combo')
  parser.add_argument('--images_dload_dir', default=osp.join('data', 'contactpose_data'),
                      help='Directory where images will be downloaded.'
                      'They will be symlinked to the appropriate location')
  args = parser.parse_args()

  downloader = ContactPoseDownloader()
  if args.type == 'grasps':
    downloader.download_grasps()
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
  for p_num, intent in product(nums, intents):
    p_id = 'full{:d}_{:s}'.format(p_num, intent)
    print('####### {:s} #######'.format(p_id))
    if args.type == 'contact_maps':
      downloader.download_contact_maps(p_num, intent)
    elif args.type == 'images':
      include_objects = args.object_names
      if include_objects is not None:
        include_objects = include_objects.split(',')
      downloader.download_images(p_num, intent, args.images_dload_dir,
                                 include_objects=include_objects)
    else:
      raise NotImplementedError
