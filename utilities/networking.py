import datetime
try:
  import dropbox
  DROPBOX_FOUND = True
except ImportError:
  DROPBOX_FOUND = False
import json
import math
import os
import random
import requests
from requests.exceptions import ConnectionError
import time
from tqdm.autonotebook import tqdm

osp = os.path
if DROPBOX_FOUND:
  dropbox_app_key = os.environ.get('DROPBOX_APP_KEY')

with open(osp.join('data', 'proxies.json'), 'r') as f:
  proxies = json.load(f)
  if ('https' not in proxies) or (proxies['https'] is None):
    proxies = None


def exponential_backoff(n, max_backoff=64.0):
  t = math.pow(2.0, n)
  t += (random.randint(0, 1000)) / 1000.0
  t = min(t, max_backoff)
  return t


def upload_dropbox(lfilename, dfilename, max_tries=7):
  """
  Upload local file lfilename to dropbox location dfilename
  Implements exponential backoff
  """
  if not DROPBOX_FOUND:
    print('Dropbox API not found')
    return False
  dbx = dropbox.Dropbox(dropbox_app_key)
  ddir, _ = osp.split(dfilename)
  ddir_exists = True
  try:
    dbx.files_get_metadata(ddir)
  except dropbox.exceptions.ApiError as err:
    ddir_exists = False
  if not ddir_exists:
    try:
      dbx.files_create_folder(ddir)
    except dropbox.exceptions.ApiError as err:
      print('*** API error', err)
      dbx.close()
      return False
  mtime = osp.getmtime(lfilename)
  with open(lfilename, 'rb') as f:
    ldata = f.read()
  upload_tries = 0
  while upload_tries < max_tries:
    try:
      res = dbx.files_upload(
          ldata, dfilename,
          dropbox.files.WriteMode.overwrite,
          client_modified=datetime.datetime(*time.gmtime(mtime)[:6]),
          mute=True)
      print('uploaded as', res.name.encode('utf8'))
      dbx.close()
      return True
    except dropbox.exceptions.ApiError as err:
      print('*** API error', err)
      dbx.close()
      return False
    except ConnectionError as err:
      t = exponential_backoff(upload_tries)
      print('*** Requests Connection error, sleeping for {:f} s'.format(t), err)
      time.sleep(t)
      upload_tries += 1
  print('*** Max upload tries exceeded')
  dbx.close()
  return False


def download_url(url, filename, progress=True, max_tries=7):
  """
  Download file from a URL to filename, optionally
  displaying progress bar with tqdm
  Implements exponential backoff
  """
  tries = 0
  while tries < max_tries:
    done = download_url_once(url, filename, progress)
    if done:
      return True
    else:
      t = exponential_backoff(tries)
      print('*** Sleeping for {:f} s'.format(t))
      time.sleep(t)
      tries += 1
  print('*** Max download tries exceeded')
  return False


def download_url_once(url, filename, progress=True):
  """
  Download file from a URL to filename, optionally
  displaying progress bar with tqdm
  taken from https://stackoverflow.com/a/37573701
  """
  # Streaming, so we can iterate over the response.
  try:
    r = requests.get(url, stream=True, proxies=proxies)
  except ConnectionError as err:
    print(err)
    return False
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
      except ConnectionError as err:
        done = False
        print(err)
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
