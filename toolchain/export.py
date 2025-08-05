import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.experimental import attempt_load
from models.yolo import DetectionModel, RN_DualDDetect
from utils.general import (LOGGER, Profile, check_img_size, check_requirements,
                           check_yaml, colorstr, file_size, get_default_args, print_args, url2file)
from utils.torch_utils import select_device, smart_inference_mode

def export_formats():
    x = [['ONNX', 'onnx', '.onnx', True, True]]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])

def try_export(inner_func):
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        prefix = inner_args['prefix']
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f'{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)')
            return f, model
        except Exception as e:
            LOGGER.info(f'{prefix} export failure ❌ {dt.t:.1f}s: {e}')
            return None, None

    return outer_func

@try_export
def export_onnx(model, im, file, opset, dynamic, simplify, prefix=colorstr('ONNX:')):
    check_requirements('onnx')
    import onnx

    LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
    f = file.with_suffix('.onnx')

    output_names = ['output0']
    if dynamic:
        dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}
        if isinstance(model, DetectionModel):
            dynamic['output0'] = {0: 'batch', 1: 'anchors'}

    torch.onnx.export(
        model.cpu() if dynamic else model,
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=dynamic or None)

    model_onnx = onnx.load(f)
    onnx.checker.check_model(model_onnx)

    d = {'stride': int(max(model.stride)), 'names': model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)

    if simplify:
        try:
            cuda = torch.cuda.is_available()
            check_requirements(('onnxruntime-gpu' if cuda else 'onnxruntime', 'onnx-simplifier>=0.4.1'))
            import onnxsim

            LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'assert check failed'
            onnx.save(model_onnx, f)
        except Exception as e:
            LOGGER.info(f'{prefix} simplifier failure: {e}')
    return f, model_onnx


@smart_inference_mode()
def run(
        data=ROOT / 'data/coco128.yaml',
        weights=ROOT / 'yolo.pt',
        imgsz=(640, 640),
        batch_size=1,
        device='cpu',
        include=('onnx',),
        half=False,
        inplace=False,
        dynamic=False,
        simplify=False,
        opset=19,
):
    t = time.time()
    include = [x.lower() for x in include]
    file = Path(url2file(weights) if str(weights).startswith(('http:/', 'https:/')) else weights)

    device = select_device(device)
    if half:
        assert device.type != 'cpu', '--half only compatible with GPU export, i.e. use --device 0'
        assert not dynamic, '--half not compatible with --dynamic'
    
    # Load model
    model = attempt_load(weights, device=device, inplace=True, fuse=True)
    
    gs = int(max(model.stride))
    imgsz = [check_img_size(x, gs) for x in imgsz]
    im = torch.zeros(batch_size, 3, *imgsz).to(device)

    # Update model for export
    model.eval()
    for k, m in model.named_modules():
        # Correctly identify our custom detection head
        if isinstance(m, RN_DualDDetect):
            m.inplace = inplace
            m.dynamic = dynamic
            m.export = True

    # Warmup
    for _ in range(2):
        y = model(im)
    if half:
        im, model = im.half(), model.half()
    
    shape = (y[0] if isinstance(y, (tuple, list)) else y).shape
    LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")

    # Export
    if 'onnx' in include:
        export_onnx(model, im, file, opset, dynamic, simplify)

    # Log results
    LOGGER.info(f'\nExport complete ({time.time() - t:.1f}s)')
    LOGGER.info(f"Results saved to {colorstr('bold', file.parent.resolve())}")
    LOGGER.info(f"Visualize:       https://netron.app")
    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default=ROOT / 'yolo.pt', help='model.pt path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set Detect() inplace=True')
    parser.add_argument('--dynamic',  default=False, action='store_true', help='ONNX: dynamic axes')
    parser.add_argument('--simplify', default=True, action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=19, help='ONNX: opset version')
    parser.add_argument('--include', nargs='+', default=['onnx'], help='Export format')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)