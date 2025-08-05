import logging
import os
import subprocess
import urllib
from pathlib import Path

import requests
import torch

def is_url(url, check=True):
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])
        return (urllib.request.urlopen(url).getcode() == 200) if check else True
    except (AssertionError, urllib.request.HTTPError):
        return False

def gsutil_getsize(url=''):
    s = subprocess.check_output(f'gsutil du {url}', shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0

def url_getsize(url='https://ultralytics.com/images/bus.jpg'):
    response = requests.head(url, allow_redirects=True)
    return int(response.headers.get('content-length', -1))

def safe_download(file, url, url2=None, min_bytes=1E0, error_msg=''):
    from utils.general import LOGGER

    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:
        LOGGER.info(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file), progress=LOGGER.level <= logging.INFO)
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg
    except Exception as e:
        if file.exists():
            file.unlink()
        LOGGER.info(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        os.system(f"curl -# -L '{url2 or url}' -o '{file}' --retry 3 -C -")
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:
            if file.exists():
                file.unlink()
            LOGGER.info(f"ERROR: {assert_msg}\n{error_msg}")
        LOGGER.info('')

def attempt_download(file, repo='ultralytics/yolov5', release='v7.0'):
    from utils.general import LOGGER

    def github_assets(repository, version='latest'):
        if version != 'latest':
            version = f'tags/{version}'
        response = requests.get(f'https://api.github.com/repos/{repository}/releases/{version}').json()
        return response['tag_name'], [x['name'] for x in response['assets']]

    file = Path(str(file).strip().replace("'", ''))
    if not file.exists():
        name = Path(urllib.parse.unquote(str(file))).name
        if str(file).startswith(('http:/', 'https:/')):
            url = str(file).replace(':/', '://')
            file = name.split('?')[0]
            if Path(file).is_file():
                LOGGER.info(f'Found {url} locally at {file}')
            else:
                safe_download(file=file, url=url, min_bytes=1E5)
            return file

        assets = [f'yolov5{size}{suffix}.pt' for size in 'nsmlx' for suffix in ('', '6', '-cls', '-seg')]
        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)
            except Exception:
                try:
                    tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
                except Exception:
                    tag = release

        file.parent.mkdir(parents=True, exist_ok=True)
        if name in assets:
            url3 = 'https://drive.google.com/drive/folders/1EFQTEUeXWSFww0luse2jB9M1QNZQGwNl'
            safe_download(
                file,
                url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
                min_bytes=1E5,
                error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/{tag} or {url3}')

    return str(file)
