# Copyright (c) Facebook, Inc. and its affiliates.
# Code by Samarth Brahmbhatt

import init_paths
from scripts.download_data import ContactPoseDownloader
import ffmpeg
import os
import shutil
import json
import itertools
from multiprocessing import Pool
import argparse
from functools import partial
import utilities.networking as nutils

osp = os.path
intents = ('use', 'handoff')
with open(osp.join('data', 'object_names.txt'), 'r') as f:
  object_names = [l.strip() for l in f]
# object_names = ('bowl', )
with open(osp.join('data', 'urls.json'), 'r') as f:
  urls = json.load(f)
urls = urls['images']

video_params = {
  'color': {
    'ffmpeg_kwargs': dict(pix_fmt='yuv420p', vcodec='libx264', crf=0),
    'ext': 'mp4',
    'valid': True,
  },
  'depth': {
    'ffmpeg_kwargs': dict(pix_fmt='gray16le', vcodec='ffv1'),
    'ext': 'mkv',
    'valid': False,  # compression not working losslessly right now, so skip
  },
}


def produce_worker(task, ffmpeg_path):
  try:
    p_num, intent, object_name = task
    p_id = 'full{:d}_{:s}'.format(p_num, intent)
    dload_dir=osp.join('data', 'contactpose_data')
    data_dir = osp.join(dload_dir, p_id, object_name, 'images_full')

    # download
    downloader = ContactPoseDownloader()
    if osp.isdir(data_dir):
      shutil.rmtree(data_dir)
      print('Deleted {:s}'.format(data_dir))
    downloader.download_images(p_num, intent, dload_dir,
                               include_objects=(object_name,))
    if not osp.isdir(data_dir):
      print('Could not download {:s} {:s}'.format(p_id, object_name))
      # check if the data actually exists
      if object_name in urls[p_id]:
        return False
      else:
        print('But that is OK because underlying data does not exist')
        return True
    
    # process
    for camera_position in ('left', 'right', 'middle'):
      camera_name = 'kinect2_{:s}'.format(camera_position)
      this_data_dir = osp.join(data_dir, camera_name)
      if not osp.isdir(this_data_dir):
        print('{:s} does not have {:s} camera'.format(this_data_dir, camera_position))
        continue
      for mode, params in video_params.items():
        if not params['valid']:
          shutil.rmtree(osp.join(this_data_dir, mode))
          continue
        # video encoding
        output_filename = osp.join(this_data_dir,
                                  '{:s}.{:s}'.format(mode, params['ext']))
        (
            ffmpeg
            .input(osp.join(this_data_dir, mode, 'frame%03d.png'), framerate=30)
            .output(output_filename, **params['ffmpeg_kwargs'])
            .overwrite_output()
            .run(cmd=ffmpeg_path)
        )
        print('{:s} written'.format(output_filename), flush=True)
        shutil.rmtree(osp.join(this_data_dir, mode))
        # upload
        dropbox_path = osp.join('/', 'contactpose',
                                'videos_full',
                                p_id, object_name, mode,
                                '{:s}.mp4'.format(camera_name))
        if not nutils.upload_dropbox(output_filename, dropbox_path):
          return False
    return True
  except Exception as e:
    print('Error somewhere in ', task)
    print(str(e))
    return False


def produce(p_nums, cleanup=False, parallel=True, ffmpeg_path='ffmpeg'):
  if cleanup:
    print('#### Cleanup mode ####')
    filename = osp.join('status.json')
    with open(filename, 'r') as f:
      status = json.load(f)
    tasks = []
    for task,done in status.items():
      if done:
        continue
      task = task.split('_')
      p_num = int(task[0][4:])
      intent = task[1]
      object_name = '_'.join(task[2:])
      tasks.append((p_num, intent, object_name))
    print('Found {:d} cleanup items'.format(len(tasks)))
  else:
    tasks = list(itertools.product(p_nums, intents, object_names))
  
  worker = partial(produce_worker, ffmpeg_path=ffmpeg_path)
  if parallel:
    p = Pool(len(object_names))
    dones = p.map(worker, tasks)
    p.close()
    p.join()
  else:
    dones = map(worker, tasks)
  
  filename = osp.join('status.json')
  d = {}
  if osp.isfile(filename):
    with open(filename, 'r') as f:
      d = json.load(f)
  for task, done in zip(tasks, dones):
    d['full{:d}_{:s}_{:s}'.format(*task)] = done
  with open(filename, 'w') as f:
    json.dump(d, f, indent=4, separators=(', ', ': '))
  print('{:s} updated'.format(filename))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-p', type=int, required=True)
  parser.add_argument('--cleanup', action='store_true')
  parser.add_argument('--no_parallel', action='store_false', dest='parallel')
  parser.add_argument('--ffmpeg_path', default='ffmpeg')
  args = parser.parse_args()
  produce((args.p, ), cleanup=args.cleanup, parallel=args.parallel)
