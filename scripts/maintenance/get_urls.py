import requests
import json
from copy import deepcopy
import os

osp = os.path

data_template = {
  'path': '/contactpose/videos_full/{:s}/{:s}/color',
  'settings': {
    'requested_visibility': 'public',
    'audience': 'public',
    'access': 'viewer'
  }
}


def get_url(p_id, object_name):
  headers = {
    'Authorization': 'Bearer {:s}'.format(os.environ['DROPBOX_APP_KEY']),
    'Content-Type': 'application/json',
  }
  d = deepcopy(data_template)
  d['path'] = d['path'].format(p_id, object_name)
  filename = '/tmp/tmpurl.json'
  with open(filename, 'w') as f:
    json.dump(d, f)
  r = requests.post('https://api.dropboxapi.com/2/sharing/create_shared_link_with_settings', data=open(filename), headers=headers)
  if r.status_code != 200:
    print('Unsuccessful, return status = {:d}'.format(r.status_code))
    return
  url = r.json()['url']
  url = url.replace('dl=0', 'dl=1')
  filename = osp.join('data', 'urls.json')
  with open(filename, 'r') as f:
    d = json.load(f)
  if p_id not in d['videos']['color']:
    d['videos']['color'][p_id] = {}
  d['videos']['color'][p_id][object_name] = url
  with open(filename, 'w') as f:
    json.dump(d, f, indent=4, separators=(', ', ': '))
  # print('{:s} updated'.format(filename))



if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--p_id', required=True)
  args = parser.parse_args()

  with open(osp.join('data', 'urls.json'), 'r') as f:
    d = json.load(f)
  object_names = d['images'][args.p_id]

  print('#########', args.p_id)
  for object_name in object_names:
    print(object_name)
    get_url(args.p_id, object_name)