import init_paths
import dropbox
import json
from requests.exceptions import ConnectionError
import os
from utilities.dataset import get_object_names

osp = os.path
dbx = dropbox.Dropbox(os.environ['DROPBOX_APP_KEY'])

def move(p_num, intent, object_name):
  p_id = 'full{:d}_{:s}'.format(p_num, intent)
  opath = osp.join('/', 'contactpose', 'videos_full', p_id, object_name)
  dpath = osp.join(opath, 'color')
  try:
    dbx.files_create_folder(dpath)
  except dropbox.exceptions.ApiError as err:
    print('*** API error', err)
    dbx.close()
    return
  print('{:s} created'.format(dpath))
  for camera_name in ('kinect2_left', 'kinect2_right', 'kinect2_middle'):
    src = osp.join(opath, '{:s}_color.mp4'.format(camera_name))
    file_exists = True
    try:
      dbx.files_get_metadata(src)
    except dropbox.exceptions.ApiError as err:
      file_exists = False
      print('{:s} does not exist'.format(src))
    if not file_exists:
      continue
    dst = osp.join(dpath, '{:s}.mp4'.format(camera_name))
    try:
      dbx.files_move(src, dst)
    except dropbox.exceptions.ApiError as err:
      print('*** API error moving {:s} -> {:s}'.format(src, dst), err)
    print('Moved {:s} -> {:s}'.format(src, dst))


if __name__ == '__main__':
  p_num = 5
  for intent in ('use', 'handoff'):
    p_id = 'full{:d}_{:s}'.format(p_num, intent)
    with open(osp.join('data', 'object_names.txt'), 'r') as f:
      object_names = [o.strip() for o in f]
    with open(osp.join('data', 'urls.json'), 'r') as f:
      urls = json.load(f)
    object_names = [o for o in object_names if o in urls['images'][p_id]]
    for object_name in object_names:
      move(p_num, intent, object_name)