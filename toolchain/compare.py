import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import onnx
import onnxruntime
import shutil
import yaml
from onnx import shape_inference
from onnxruntime.quantization import QuantType, quantize_static
from scipy.spatial.distance import cosine
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] if str(FILE.parents[1]) in sys.path else FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from utils.dataloaders import LoadImages
from utils.general import LOGGER, colorstr, print_args, check_img_size
from utils.torch_utils import select_device

class ImageCalibrator:
    def __init__(self, calib_dir_path, input_name, imgsz=(640, 640), stride=32):
        self.input_name = input_name
        self.dataloader = LoadImages(str(calib_dir_path), img_size=imgsz, stride=stride, auto=False)
        self.iterator = iter(self.dataloader)
        LOGGER.info(f"Using {len(self.dataloader)} images for calibration...")

    def get_next(self):
        try:
            _, im, _, _, _ = next(self.iterator)
            im = im.astype(np.float32) / 255.0
            if len(im.shape) == 3:
                im = np.expand_dims(im, 0)
            return {self.input_name: im}
        except StopIteration:
            return None

def quantize_model(opt):
    model_input_path = Path(opt.weights)
    model_output_path = model_input_path.with_stem(f"{model_input_path.stem}_int8")

    LOGGER.info(f"Starting INT8 static quantization for {model_input_path}...")

    data_yaml_path = Path(opt.data)
    with open(data_yaml_path, errors="ignore") as f:
        data_dict = yaml.safe_load(f)

    dataset_root = (data_yaml_path.parent / ".." / data_dict['path']).resolve()

    calib_data_list_path = dataset_root / data_dict['train']

    with open(calib_data_list_path) as f:
        calib_files = [line.strip() for line in f][:200]

    calib_image_paths = [str(dataset_root / p) for p in calib_files]
    LOGGER.info(f"Found {len(calib_image_paths)} images from the training set for calibration.")

    calib_dir = Path('./calib_images_temp')
    calib_dir.mkdir(exist_ok=True)

    LOGGER.info(
        f"Creating temporary directory with symlinks for calibration: {calib_dir.resolve()}"
    )
    for img_path_str in calib_image_paths:
        img_path = Path(img_path_str)
        try:
            os.symlink(img_path.resolve(), calib_dir / img_path.name)
        except FileExistsError:
            pass

    try:
        temp_session = onnxruntime.InferenceSession(
            str(model_input_path), providers=["CPUExecutionProvider"]
        )
        input_name = temp_session.get_inputs()[0].name
        model_meta = temp_session.get_modelmeta().custom_metadata_map
        stride = int(model_meta.get("stride", 32))
        del temp_session

        calibrator = ImageCalibrator(
            calib_dir_path=calib_dir, input_name=input_name, imgsz=opt.imgsz, stride=stride
        )

        nodes_to_exclude = []
        model = onnx.load(model_input_path)
        # for node in model.graph.node:
        #     if node.op_type in ['HardSwish', 'Softmax']:
        #         nodes_to_exclude.append(node.name)
        if nodes_to_exclude:
            LOGGER.info(f"Excluding {len(nodes_to_exclude)} sensitive nodes from quantization.")

        quantize_static(
            model_input=model_input_path,
            model_output=model_output_path,
            calibration_data_reader=calibrator,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=opt.per_channel,
            nodes_to_exclude=nodes_to_exclude,
        )
        LOGGER.info(f"Successfully created INT8 model: {model_output_path}")
    finally:
        shutil.rmtree(calib_dir)
        LOGGER.info(f"Cleaned up temporary calibration directory.")

    return str(model_output_path)

def create_debug_session(model_path, providers):
    model = onnx.load(model_path)
    model = shape_inference.infer_shapes(model)

    tensor_type_map = {}
    for tensor in model.graph.initializer:
        tensor_type_map[tensor.name] = tensor.data_type
    for tensor_info in list(model.graph.input) + list(model.graph.value_info
                                                     ) + list(model.graph.output):
        if tensor_info.name in tensor_type_map:
            continue
        tensor_type_map[tensor_info.name] = tensor_info.type.tensor_type.elem_type

    all_tensor_names = {tensor.name for tensor in model.graph.initializer}
    for node in model.graph.node:
        all_tensor_names.update(node.input)
        all_tensor_names.update(node.output)

    while len(model.graph.output):
        model.graph.output.pop()

    for name in sorted(list(all_tensor_names)):
        if name and name in tensor_type_map:
            data_type = tensor_type_map.get(name)
            output_tensor_info = onnx.helper.make_tensor_value_info(name, data_type, None)
            model.graph.output.append(output_tensor_info)

    session = onnxruntime.InferenceSession(model.SerializeToString(), providers=providers)
    output_names = [output.name for output in session.get_outputs()]
    return session, output_names

def run_and_collect_outputs(session, input_name, input_image, output_names):
    outputs = session.run(output_names, {input_name: input_image})
    return {name: out for name, out in zip(output_names, outputs)}

def compare_models(opt):
    fp32_path = opt.weights
    int8_path_obj = Path(fp32_path).with_stem(f"{Path(fp32_path).stem}_int8")

    if opt.quantize or not int8_path_obj.exists():
        if not opt.data:
            raise ValueError(
                "INT8 quantization requires a --data argument for the calibration dataset."
            )
        int8_path = quantize_model(opt)
    else:
        int8_path = str(int8_path_obj)
        LOGGER.info(f"Found existing INT8 model at {int8_path}, skipping quantization.")

    LOGGER.info(f"\nComparing FP32 model: {fp32_path}")
    LOGGER.info(f"With INT8 model:      {int8_path}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"
                ] if select_device(opt.device).type != "cpu" else ["CPUExecutionProvider"]

    LOGGER.info("Creating debug sessions (this may take a moment)...")
    session_fp32, fp32_output_names = create_debug_session(fp32_path, providers)
    session_int8, int8_output_names = create_debug_session(int8_path, providers)

    input_name = session_fp32.get_inputs()[0].name

    LOGGER.info(f"Loading image {opt.source} for comparison...")
    dataloader = LoadImages(opt.source, img_size=opt.imgsz, auto=False)
    _, im, _, _, _ = next(iter(dataloader))

    im = im.astype(np.float32) / 255.0
    if len(im.shape) == 3:
        im = np.expand_dims(im, 0)

    LOGGER.info("Running models to collect intermediate outputs...")
    outputs_fp32 = run_and_collect_outputs(session_fp32, input_name, im, fp32_output_names)
    outputs_int8 = run_and_collect_outputs(session_int8, input_name, im, int8_output_names)

    print("\n" + "=" * 45 + " ONNX LAYER-BY-LAYER COMPARISON " + "=" * 45)
    print(f"{'Layer (Tensor Name)':<75} {'Cosine Sim':>15s} {'MAE':>15s}")
    print('-' * 110)

    common_names = sorted(list(set(outputs_fp32.keys()) & set(outputs_int8.keys())))

    for name in common_names:
        fp32_val, int8_val = outputs_fp32.get(name), outputs_int8.get(name)

        if fp32_val is None or int8_val is None:
            continue
        if fp32_val.shape != int8_val.shape:
            LOGGER.warning(
                f"Shape mismatch for tensor '{name}': FP32={fp32_val.shape}, INT8={int8_val.shape}"
            )
            continue

        flat_fp32 = fp32_val.flatten()
        flat_int8 = int8_val.flatten().astype(np.float32)

        if flat_fp32.size == 0 or flat_int8.size == 0:
            continue

        cos_sim = 1 - cosine(flat_fp32,
                             flat_int8) if np.any(flat_fp32) or np.any(flat_int8) else 1.0
        mae = np.mean(np.abs(flat_fp32 - flat_int8))

        color_start = "\033[91m" if cos_sim < 0.95 else ""
        color_end = "\033[0m" if cos_sim < 0.95 else ""

        print(f"{name:<75} {color_start}{cos_sim:15.6f}{color_end} {mae:15.6f}")

def main(opt):
    compare_models(opt)

def parse_opt():
    parser = argparse.ArgumentParser(
        description="ONNX FP32 vs. INT8 Layer-by-Layer Comparison Tool"
    )
    parser.add_argument("--weights", type=str, required=True, help="Path to the FP32 ONNX model.")
    parser.add_argument(
        "--source", type=str, required=True, help="Path to a single image for comparison."
    )
    parser.add_argument("--data", type=str, help="Path to dataset.yaml for INT8 calibration.")
    parser.add_argument(
        "--imgsz", "--img-size", nargs="+", type=int, default=[640], help="Inference size h,w."
    )
    parser.add_argument("--device", default="cpu", help="CUDA device, i.e., 0 or cpu.")
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Force re-quantization even if an INT8 model exists."
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Enable per-channel quantization for INT8 conversion."
    )

    opt = parser.parse_args()
    opt.imgsz = check_img_size(opt.imgsz)

    if len(opt.imgsz) == 1:
        opt.imgsz *= 2

    print_args(vars(opt))
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
