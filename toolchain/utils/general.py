import contextlib
import glob
import inspect
import logging
import logging.config
import math
import os
import platform
import random
import re
import signal
import sys
import time
import urllib
from copy import deepcopy
from datetime import datetime
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from tarfile import is_tarfile
from typing import Optional
from zipfile import ZipFile, is_zipfile

import cv2
import IPython
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml

from utils import TryExcept, emojis
from utils.downloads import gsutil_getsize
from utils.metrics import box_iou, fitness

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  
RANK = int(os.getenv('RANK', -1))

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  
DATASETS_DIR = Path(os.getenv('YOLOv5_DATASETS_DIR', ROOT.parent / 'datasets'))  
AUTOINSTALL = str(os.getenv('YOLOv5_AUTOINSTALL', True)).lower() == 'true'  
VERBOSE = str(os.getenv('YOLOv5_VERBOSE', True)).lower() == 'true'  
TQDM_BAR_FORMAT = '{l_bar}{bar:10}| {n_fmt}/{total_fmt} {elapsed}'  
FONT = 'Arial.ttf'  

torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)  
os.environ['OMP_NUM_THREADS'] = '1' if platform.system() == 'darwin' else str(NUM_THREADS)

def is_ascii(s=''):
    s = str(s)  
    return len(s.encode().decode('ascii', 'ignore')) == len(s)

def is_chinese(s='人工智能'):
    return bool(re.search('[\u4e00-\u9fff]', str(s)))

def is_colab():
    return 'google.colab' in sys.modules

def is_notebook():
    ipython_type = str(type(IPython.get_ipython()))
    return 'colab' in ipython_type or 'zmqshell' in ipython_type

def is_kaggle():
    return os.environ.get('PWD') == '/kaggle/working' and os.environ.get('KAGGLE_URL_BASE') == 'https://www.kaggle.com'

def is_docker() -> bool:
    """Check if the process runs inside a docker container."""
    if Path("/.dockerenv").exists():
        return True
    try:  
        with open("/proc/self/cgroup") as file:
            return any("docker" in line for line in file)
    except OSError:
        return False

def is_writeable(dir, test=False):
    if not test:
        return os.access(dir, os.W_OK)  
    file = Path(dir) / 'tmp.txt'
    try:
        with open(file, 'w'):  
            pass
        file.unlink()  
        return True
    except OSError:
        return False

LOGGING_NAME = "yolov5"

def set_logging(name=LOGGING_NAME, verbose=True):
    rank = int(os.getenv('RANK', -1))  
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            name: {
                "format": "%(message)s"}},
        "handlers": {
            name: {
                "class": "logging.StreamHandler",
                "formatter": name,
                "level": level,}},
        "loggers": {
            name: {
                "level": level,
                "handlers": [name],
                "propagate": False,}}})

set_logging(LOGGING_NAME)  
LOGGER = logging.getLogger(LOGGING_NAME)  
if platform.system() == 'Windows':
    for fn in LOGGER.info, LOGGER.warning:
        setattr(LOGGER, fn.__name__, lambda x: fn(emojis(x)))

def user_config_dir(dir='Ultralytics', env_var='YOLOV5_CONFIG_DIR'):
    env = os.getenv(env_var)
    if env:
        path = Path(env)  
    else:
        cfg = {'Windows': 'AppData/Roaming', 'Linux': '.config', 'Darwin': 'Library/Application Support'}  
        path = Path.home() / cfg.get(platform.system(), '')  
        path = (path if is_writeable(path) else Path('/tmp')) / dir  
    path.mkdir(exist_ok=True)  
    return path

CONFIG_DIR = user_config_dir()

class Profile(contextlib.ContextDecorator):
    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  
        self.t += self.dt  

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()

class Timeout(contextlib.ContextDecorator):
    def __init__(self, seconds, *, timeout_msg='', suppress_timeout_errors=True):
        self.seconds = int(seconds)
        self.timeout_message = timeout_msg
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        if platform.system() != 'Windows':  
            signal.signal(signal.SIGALRM, self._timeout_handler)  
            signal.alarm(self.seconds)  

    def __exit__(self, exc_type, exc_val, exc_tb):
        if platform.system() != 'Windows':
            signal.alarm(0)  
            if self.suppress and exc_type is TimeoutError:  
                return True

class WorkingDirectory(contextlib.ContextDecorator):
    def __init__(self, new_dir):
        self.dir = new_dir  
        self.cwd = Path.cwd().resolve()  

    def __enter__(self):
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.cwd)

def methods(instance):
    return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith("__")]

def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    x = inspect.currentframe().f_back  
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    LOGGER.info(colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))

def init_seeds(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    
    if deterministic and check_version(torch.__version__, '1.12.0'):  
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)

def intersect_dicts(da, db, exclude=()):
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

def get_default_args(func):
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}

def get_latest_run(search_dir='.'):
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''

def file_age(path=__file__):
    dt = (datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime))  
    return dt.days

def file_date(path=__file__):
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'

def file_size(path):
    mb = 1 << 20  
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    else:
        return 0.0

def check_online():
    import socket

    def run_once():
        try:
            socket.create_connection(("1.1.1.1", 443), 5)  
            return True
        except OSError:
            return False

    return run_once() or run_once()

def git_describe(path=ROOT):
    try:
        assert (Path(path) / '.git').is_dir()
        return check_output(f'git -C {path} describe --tags --long --always', shell=True).decode()[:-1]
    except Exception:
        return ''

@TryExcept()
@WorkingDirectory(ROOT)
def check_git_status(repo='WongKinYiu/yolov9', branch='main'):
    url = f'https://github.com/{repo}'
    msg = f', for updates see {url}'
    s = colorstr('github: ')  
    assert Path('.git').exists(), s + 'skipping check (not a git repository)' + msg
    assert check_online(), s + 'skipping check (offline)' + msg

    splits = re.split(pattern=r'\s', string=check_output('git remote -v', shell=True).decode())
    matches = [repo in s for s in splits]
    if any(matches):
        remote = splits[matches.index(True) - 1]
    else:
        remote = 'ultralytics'
        check_output(f'git remote add {remote} {url}', shell=True)
    check_output(f'git fetch {remote}', shell=True, timeout=5)  
    local_branch = check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()  
    n = int(check_output(f'git rev-list {local_branch}..{remote}/{branch} --count', shell=True))  
    if n > 0:
        pull = 'git pull' if remote == 'origin' else f'git pull {remote} {branch}'
        s += f"⚠️ YOLO is out of date by {n} commit{'s' * (n > 1)}. Use `{pull}` or `git clone {url}` to update."
    else:
        s += f'up to date with {url} ✅'
    LOGGER.info(s)

@WorkingDirectory(ROOT)
def check_git_info(path='.'):
    check_requirements('gitpython')
    import git
    try:
        repo = git.Repo(path)
        remote = repo.remotes.origin.url.replace('.git', '')  
        commit = repo.head.commit.hexsha  
        try:
            branch = repo.active_branch.name  
        except TypeError:  
            branch = None  
        return {'remote': remote, 'branch': branch, 'commit': commit}
    except git.exc.InvalidGitRepositoryError:  
        return {'remote': None, 'branch': None, 'commit': None}

def check_python(minimum='3.7.0'):
    check_version(platform.python_version(), minimum, name='Python ', hard=True)

def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  
    s = f'WARNING ⚠️ {name}{minimum} is required by YOLO, but {name}{current} is currently installed'  
    if hard:
        assert result, emojis(s)  
    if verbose and not result:
        LOGGER.warning(s)
    return result

@TryExcept()
def check_requirements(requirements=ROOT / 'requirements.txt', exclude=(), install=True, cmds=''):
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()  
    if isinstance(requirements, Path):  
        file = requirements.resolve()
        assert file.exists(), f"{prefix} {file} not found, check failed."
        with file.open() as f:
            requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(f) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]

    s = ''
    n = 0
    for r in requirements:
        try:
            pkg.require(r)
        except (pkg.VersionConflict, pkg.DistributionNotFound):  
            s += f'"{r}" '
            n += 1

    if s and install and AUTOINSTALL:  
        LOGGER.info(f"{prefix} YOLO requirement{'s' * (n > 1)} {s}not found, attempting AutoUpdate...")
        try:
            LOGGER.info(check_output(f'pip install {s} {cmds}', shell=True).decode())
            source = file if 'file' in locals() else requirements
            s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
                f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
            LOGGER.info(s)
        except Exception as e:
            LOGGER.warning(f'{prefix} ❌ {e}')

def check_img_size(imgsz, s=32, floor=0):
    if isinstance(imgsz, int):  
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  
        imgsz = list(imgsz)  
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f'WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size

def check_imshow(warn=False):
    try:
        assert not is_notebook()
        assert not is_docker()
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f'WARNING ⚠️ Environment does not support cv2.imshow() or PIL Image.show()\n{e}')
        return False

def check_suffix(file='yolo.pt', suffix=('.pt',), msg=''):
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"

def check_yaml(file, suffix=('.yaml', '.yml')):
    return check_file(file, suffix)

def check_file(file, suffix=''):
    check_suffix(file, suffix)  
    file = str(file)  
    if os.path.isfile(file) or not file:  
        return file
    elif file.startswith(('http:/', 'https:/')):  
        url = file  
        file = Path(urllib.parse.unquote(file).split('?')[0]).name  
        if os.path.isfile(file):
            LOGGER.info(f'Found {url} locally at {file}')  
        else:
            LOGGER.info(f'Downloading {url} to {file}...')
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  
        return file
    elif file.startswith('clearml://'):  
        assert 'clearml' in sys.modules, "ClearML is not installed, so cannot use ClearML dataset. Try running 'pip install clearml'."
        return file
    else:  
        files = []
        for d in 'data', 'models', 'utils':  
            files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True))  
        assert len(files), f'File not found: {file}'  
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  
        return files[0]

def check_font(font=FONT, progress=False):
    font = Path(font)
    file = CONFIG_DIR / font.name
    if not font.exists() and not file.exists():
        url = f'https://ultralytics.com/assets/{font.name}'
        LOGGER.info(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file), progress=progress)

def check_dataset(data, autodownload=True):
    extract_dir = ''
    if isinstance(data, (str, Path)) and (is_zipfile(data) or is_tarfile(data)):
        download(data, dir=f'{DATASETS_DIR}/{Path(data).stem}', unzip=True, delete=False, curl=False, threads=1)
        data = next((DATASETS_DIR / Path(data).stem).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False

    
    if isinstance(data, (str, Path)):
        data = yaml_load(data)  

    
    for k in 'train', 'val', 'names':
        assert k in data, emojis(f"data.yaml '{k}:' field missing ❌")
    if isinstance(data['names'], (list, tuple)):  
        data['names'] = dict(enumerate(data['names']))  
    assert all(isinstance(k, int) for k in data['names'].keys()), 'data.yaml names keys must be integers, i.e. 2: car'
    data['nc'] = len(data['names'])

    
    path = Path(extract_dir or data.get('path') or '')  
    if not path.is_absolute():
        path = (ROOT / path).resolve()
        data['path'] = path  
    for k in 'train', 'val', 'test':
        if data.get(k):  
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith('../'):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]
    train, val, test, s = (data.get(x) for x in ('train', 'val', 'test', 'download'))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  
        if not all(x.exists() for x in val):
            LOGGER.info('\nDataset not found ⚠️, missing paths %s' % [str(x) for x in val if not x.exists()])
            if not s or not autodownload:
                raise Exception('Dataset not found ❌')
            t = time.time()
            if s.startswith('http') and s.endswith('.zip'):  
                f = Path(s).name  
                LOGGER.info(f'Downloading {s} to {f}...')
                torch.hub.download_url_to_file(s, f)
                Path(DATASETS_DIR).mkdir(parents=True, exist_ok=True)  
                unzip_file(f, path=DATASETS_DIR)  
                Path(f).unlink()  
                r = None  
            elif s.startswith('bash '):  
                LOGGER.info(f'Running {s} ...')
                r = os.system(s)
            else:  
                r = exec(s, {'yaml': data})  
            dt = f'({round(time.time() - t, 1)}s)'
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in (0, None) else f"failure {dt} ❌"
            LOGGER.info(f"Dataset download {s}")
    check_font('Arial.ttf' if is_ascii(data['names']) else 'Arial.Unicode.ttf', progress=True)  
    return data

def check_amp(model):
    from models.common import AutoShape, DetectMultiBackend
    def amp_allclose(model, im):
        m = AutoShape(model, verbose=False)  
        a = m(im).xywhn[0]  
        m.amp = True
        b = m(im).xywhn[0]  
        return a.shape == b.shape and torch.allclose(a, b, atol=0.1)  

    prefix = colorstr('AMP: ')
    device = next(model.parameters()).device  
    if device.type in ('cpu', 'mps'):
        return False  
    f = ROOT / 'data' / 'images' / 'bus.jpg'  
    im = f if f.exists() else 'https://ultralytics.com/images/bus.jpg' if check_online() else np.ones((640, 640, 3))
    try:
        LOGGER.info(f'{prefix}checks passed ✅')
        return True
    except Exception:
        help_url = 'https://github.com/ultralytics/yolov5/issues/7908'
        LOGGER.warning(f'{prefix}checks failed ❌, disabling Automatic Mixed Precision. See {help_url}')
        return False

def yaml_load(file='data.yaml'):
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)

def yaml_save(file='data.yaml', data={}):
    with open(file, 'w') as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)

def unzip_file(file, path=None, exclude=('.DS_Store', '__MACOSX')):
    if path is None:
        path = Path(file).parent  
    with ZipFile(file) as zipObj:
        for f in zipObj.namelist():  
            if all(x not in f for x in exclude):
                zipObj.extract(f, path=path)

def url2file(url):
    url = str(Path(url)).replace(':/', '://')  
    return Path(urllib.parse.unquote(url)).name.split('?')[0]

def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1, retry=3):
    def download_one(url, dir):
        success = True
        if os.path.isfile(url):
            f = Path(url)  
        else:  
            f = dir / Path(url).name
            LOGGER.info(f'Downloading {url} to {f}...')
            for i in range(retry + 1):
                if curl:
                    s = 'sS' if threads > 1 else ''  
                    r = os.system(
                        f'curl -# -{s}L "{url}" -o "{f}" --retry 9 -C -')  # curl download with retry, continue
                    success = r == 0
                else:
                    torch.hub.download_url_to_file(url, f, progress=threads == 1)  
                    success = f.is_file()
                if success:
                    break
                elif i < retry:
                    LOGGER.warning(f'⚠️ Download failure, retrying {i + 1}/{retry} {url}...')
                else:
                    LOGGER.warning(f'❌ Failed to download {url}...')

        if unzip and success and (f.suffix == '.gz' or is_zipfile(f) or is_tarfile(f)):
            LOGGER.info(f'Unzipping {f}...')
            if is_zipfile(f):
                unzip_file(f, dir)  
            elif is_tarfile(f):
                os.system(f'tar xf {f} --directory {f.parent}')  
            elif f.suffix == '.gz':
                os.system(f'tar xfz {f} --directory {f.parent}')  
            if delete:
                f.unlink()  

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)

def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  
    return math.ceil(x / divisor) * divisor

def clean_str(s):
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)

def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

def one_flat_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: ((1 - math.cos((x - (steps // 2)) * math.pi / (steps // 2))) / 2) * (y2 - y1) + y1 if (x > (steps // 2)) else y1

def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  
    colors = {
        'black': '\033[30m',  
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def labels_to_class_weights(labels, nc=80):
    if labels[0] is None:  
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  
    classes = labels[:, 0].astype(int)  
    weights = np.bincount(classes, minlength=nc)

    weights[weights == 0] = 1  
    weights = 1 / weights  
    weights /= weights.sum()  
    return torch.from_numpy(weights).float()

def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    class_counts = np.array([np.bincount(x[:, 0].astype(int), minlength=nc) for x in labels])
    return (class_weights.reshape(1, nc) * class_counts).sum(1)

def coco80_to_coco91_class():
    return [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  
    y[..., 2] = x[..., 2] - x[..., 0]  
    y[..., 3] = x[..., 3] - x[..., 1]  
    return y

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  
    y[..., 1] = x[..., 1] - x[..., 3] / 2  
    y[..., 2] = x[..., 0] + x[..., 2] / 2  
    y[..., 3] = x[..., 1] + x[..., 3] / 2  
    return y

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  
    return y

def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    if clip:
        clip_boxes(x, (h - eps, w - eps))  
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  
    return y

def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * x[..., 0] + padw  
    y[..., 1] = h * x[..., 1] + padh  
    return y

def segment2box(segment, width=640, height=640):
    x, y = segment.T  
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))

def segments2boxes(segments):
    boxes = []
    for s in segments:
        x, y = s.T  
        boxes.append([x.min(), y.min(), x.max(), y.max()])  
    return xyxy2xywh(np.array(boxes))

def resample_segments(segments, n=1000):
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  
    return segments

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    if ratio_pad is None:  
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  
    boxes[:, [1, 3]] -= pad[1]  
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def scale_segments(img1_shape, segments, img0_shape, ratio_pad=None, normalize=False):
    if ratio_pad is None:  
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    segments[:, 0] -= pad[0]  
    segments[:, 1] -= pad[1]  
    segments /= gain
    clip_segments(segments, img0_shape)
    if normalize:
        segments[:, 0] /= img0_shape[1]  
        segments[:, 1] /= img0_shape[0]  
    return segments

def clip_boxes(boxes, shape):
    if isinstance(boxes, torch.Tensor):  
        boxes[:, 0].clamp_(0, shape[1])  
        boxes[:, 1].clamp_(0, shape[0])  
        boxes[:, 2].clamp_(0, shape[1])  
        boxes[:, 3].clamp_(0, shape[0])  
    else:  
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])

def clip_segments(segments, shape):
    if isinstance(segments, torch.Tensor):  
        segments[:, 0].clamp_(0, shape[1])  
        segments[:, 1].clamp_(0, shape[0])  
    else:  
        segments[:, 0] = segments[:, 0].clip(0, shape[1])  
        segments[:, 1] = segments[:, 1].clip(0, shape[0])

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(prediction, (list, tuple)):  
        prediction = prediction[0]  

    device = prediction.device
    mps = 'mps' in device.type  
    if mps:  
        prediction = prediction.cpu()
    bs = prediction.shape[0]  
    nc = prediction.shape[1] - nm - 4  
    mi = 4 + nc  
    xc = prediction[:, 4:mi].amax(1) > conf_thres  

    
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    
    
    max_wh = 7680  
    max_nms = 30000  
    time_limit = 2.5 + 0.05 * bs  
    redundant = True  
    multi_label &= nc > 1  
    merge = False  

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x.T[xc[xi]]  

        
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  
            x = torch.cat((x, v), 0)

        
        if not x.shape[0]:
            continue

        
        box, cls, mask = x.split((4, nc, nm), 1)
        box = xywh2xyxy(box)  
        if multi_label:
            i, j = (cls > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        
        n = x.shape[0]  
        if not n:  
            continue
        elif n > max_nms:  
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  
        else:
            x = x[x[:, 4].argsort(descending=True)]  

        
        c = x[:, 5:6] * (0 if agnostic else max_wh)  
        boxes, scores = x[:, :4] + c, x[:, 4]  
        i = torchvision.ops.nms(boxes, scores, iou_thres)  
        if i.shape[0] > max_det:  
            i = i[:max_det]
        if merge and (1 < n < 3E3):
            iou = box_iou(boxes[i], boxes) > iou_thres  
            weights = iou * scores[None]  
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  
            if redundant:
                i = i[iou.sum(1) > 1]  

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  

    return output

def strip_optimizer(f='best.pt', s=''):
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']  
    for k in 'optimizer', 'best_fitness', 'ema', 'updates':  
        x[k] = None
    x['epoch'] = -1
    x['model'].half()  
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")

def print_mutation(keys, results, hyp, save_dir, bucket, prefix=colorstr('evolve: ')):
    evolve_csv = save_dir / 'evolve.csv'
    evolve_yaml = save_dir / 'hyp_evolve.yaml'
    keys = tuple(keys) + tuple(hyp.keys())  
    keys = tuple(x.strip() for x in keys)
    vals = results + tuple(hyp.values())
    n = len(keys)

    
    if bucket:
        url = f'gs://{bucket}/evolve.csv'
        if gsutil_getsize(url) > (evolve_csv.stat().st_size if evolve_csv.exists() else 0):
            os.system(f'gsutil cp {url} {save_dir}')  

    
    s = '' if evolve_csv.exists() else (('%20s,' * n % keys).rstrip(',') + '\n')  
    with open(evolve_csv, 'a') as f:
        f.write(s + ('%20.5g,' * n % vals).rstrip(',') + '\n')

    
    with open(evolve_yaml, 'w') as f:
        data = pd.read_csv(evolve_csv)
        data = data.rename(columns=lambda x: x.strip())  
        i = np.argmax(fitness(data.values[:, :4]))  
        generations = len(data)
        f.write('# YOLO Hyperparameter Evolution Results\n' + f'# Best generation: {i}\n' +
                f'# Last generation: {generations - 1}\n' + '# ' + ', '.join(f'{x.strip():>20s}' for x in keys[:7]) +
                '\n' + '# ' + ', '.join(f'{x:>20.5g}' for x in data.values[i, :7]) + '\n\n')
        yaml.safe_dump(data.loc[i][7:].to_dict(), f, sort_keys=False)

    
    LOGGER.info(prefix + f'{generations} generations finished, current result:\n' + prefix +
                ', '.join(f'{x.strip():>20s}' for x in keys) + '\n' + prefix + ', '.join(f'{x:20.5g}'
                                                                                         for x in vals) + '\n\n')

    if bucket:
        os.system(f'gsutil cp {evolve_csv} {evolve_yaml} gs://{bucket}')

def apply_classifier(x, model, img, im0):
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  
        if d is not None and len(d):
            d = d.clone()

            
            b = xyxy2xywh(d[:, :4])  
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  
            b[:, 2:] = b[:, 2:] * 1.3 + 30  
            d[:, :4] = xywh2xyxy(b).long()

            
            scale_boxes(img.shape[2:], d[:, :4], im0[i].shape)

            
            pred_cls1 = d[:, 5].long()
            ims = []
            for a in d:
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  

                im = im[:, :, ::-1].transpose(2, 0, 1)  
                im = np.ascontiguousarray(im, dtype=np.float32)  
                im /= 255  
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  
            x[i] = x[i][pred_cls1 == pred_cls2]  

    return x

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    path = Path(path)  
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  
            if not os.path.exists(p):  
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  

    return path

imshow_ = cv2.imshow

def imread(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, np.uint8), flags)

def imwrite(path, im):
    try:
        cv2.imencode(Path(path).suffix, im)[1].tofile(path)
        return True
    except Exception:
        return False

def imshow(path, im):
    imshow_(path.encode('unicode_escape').decode(), im)

cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow

