import argparse
import csv
import importlib.util
import json
import os
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torch.nn.parameter import UninitializedParameter

import model as Model
from util.img_read_save import img_save


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PET_EVAL_PATH = os.path.join(SCRIPT_DIR, "test-med-PET.py")
spec = importlib.util.spec_from_file_location("pet_eval_helpers", PET_EVAL_PATH)
pet_eval_helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pet_eval_helpers)

PAPER_METRICS = ("MI", "VIF", "QABF", "SSIM", "FMI_pixel", "SCD", "AG")


def count_initialized_parameters(net):
    return sum(
        param.nelement()
        for param in net.parameters()
        if not isinstance(param, UninitializedParameter)
    )


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        json_str = ""
        for line in file:
            json_str += line.split("//")[0] + "\n"
    return json.loads(json_str, object_pairs_hook=OrderedDict)


def dict_to_nonedict(value):
    if isinstance(value, dict):
        converted = {key: dict_to_nonedict(sub_value) for key, sub_value in value.items()}
        return type("NoneDict", (dict,), {"__missing__": lambda self, key: None})(**converted)
    if isinstance(value, list):
        return [dict_to_nonedict(sub_value) for sub_value in value]
    return value


def write_metric_reports(output_folder, records, metrics, efficiency_info):
    per_image_path = os.path.join(output_folder, "metrics_per_image.csv")
    summary_path = os.path.join(output_folder, "metrics_summary.csv")
    efficiency_path = os.path.join(output_folder, "metrics_efficiency.csv")

    per_image_fields = ["image", "MI", "VIF", "QABF", "SSIM", "FMI_pixel", "SCD", "AG"]

    with open(per_image_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=per_image_fields)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "mean", "std"])
        for key in PAPER_METRICS:
            values = np.asarray(metrics[key], dtype=np.float64)
            writer.writerow([key, f"{np.mean(values):.6f}", f"{np.std(values):.6f}"])

    with open(efficiency_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["item", "value"])
        for key, value in efficiency_info.items():
            writer.writerow([key, value])

    return per_image_path, summary_path, efficiency_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config/test_ct_mri.json")
    parser.add_argument("-local_rank", "--local_rank", type=int, default=0)
    parser.add_argument("-gpu", "--gpu_ids", type=str, default=None)
    parser.add_argument("--resume_state", type=str, default=None, help="Override the checkpoint path in the config.")
    parser.add_argument("--max_images", type=int, default=None, help="Evaluate only the first N images for smoke tests.")
    args = parser.parse_args()

    opt = load_config(args.config)
    if args.resume_state is not None:
        opt["path"]["resume_state"] = args.resume_state
    gpu_list = args.gpu_ids if args.gpu_ids is not None else ",".join(str(x) for x in opt["gpu_ids"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    print("export CUDA_VISIBLE_DEVICES=" + gpu_list)
    opt["distributed"] = len(gpu_list.split(",")) > 1
    opt = dict_to_nonedict(opt)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CT-MRI inference.")

    value_range = (-1, 1) if opt["datasets"]["centered"] else (0, 1)
    dataset_name = "CT-MRI"
    test_folder = os.path.join(SCRIPT_DIR, "dataset", "test", "Med", dataset_name)
    test_out_folder = os.path.join(SCRIPT_DIR, "dataset", "test_result", "MED", dataset_name, opt["name"])
    os.makedirs(test_out_folder, exist_ok=True)

    device = torch.device("cuda")
    diffusion = Model.create_model(opt, args.local_rank)
    diffusion.Fusion_net.to(device)
    diffusion.Fusion_net.eval()

    total = count_initialized_parameters(diffusion.Fusion_net)
    params_m = total / 1e6
    print("The test result of " + dataset_name + " :")
    print("Number of parameters: %.2fM" % params_m)

    metrics = OrderedDict((key, []) for key in PAPER_METRICS)
    records = []
    timings = []
    image_names = sorted(os.listdir(os.path.join(test_folder, "MRI")))
    if args.max_images is not None:
        image_names = image_names[: max(int(args.max_images), 0)]

    with torch.no_grad():
        for img_name in tqdm(image_names, desc="Processing images"):
            start_time = time.time()
            ct_pil = Image.open(os.path.join(test_folder, "CT", img_name)).convert("L")
            mri_pil = Image.open(os.path.join(test_folder, "MRI", img_name)).convert("L")

            ct_gray = np.asarray(ct_pil, dtype=np.uint8)
            mri_gray = np.asarray(mri_pil, dtype=np.uint8)
            ct_tensor = (ToTensor()(ct_pil) * (value_range[1] - value_range[0]) + value_range[0]).unsqueeze(0).to(device)
            mri_tensor = (ToTensor()(mri_pil) * (value_range[1] - value_range[0]) + value_range[0]).unsqueeze(0).to(device)
            input_tensor = torch.cat([ct_tensor, mri_tensor], dim=1)

            fused_tensor = diffusion.Fusion_net.test_Fusion(input_tensor, device)
            fused_gray = pet_eval_helpers.tensor_to_gray_u8(fused_tensor, value_range)
            img_save(fused_gray, img_name.split(".")[0], test_out_folder)

            target_h, target_w = fused_gray.shape[:2]
            ct_eval = cv2.resize(ct_gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            mri_eval = cv2.resize(mri_gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            record = OrderedDict(
                [
                    ("image", img_name),
                    ("MI", pet_eval_helpers.calculate_q_mi(ct_eval, mri_eval, fused_gray)),
                    ("VIF", pet_eval_helpers.calculate_q_vif(ct_eval, mri_eval, fused_gray)[0]),
                    ("QABF", pet_eval_helpers.calculate_qabf(ct_eval, mri_eval, fused_gray)),
                    ("SSIM", pet_eval_helpers.calculate_q_ssim(ct_eval, mri_eval, fused_gray)[0]),
                    ("FMI_pixel", pet_eval_helpers.calculate_fmi_pixel(ct_eval, mri_eval, fused_gray)),
                    ("SCD", pet_eval_helpers.calculate_q_scd(ct_eval, mri_eval, fused_gray)),
                    ("AG", pet_eval_helpers.calculate_ag(fused_gray)),
                ]
            )
            proc_time = time.time() - start_time
            records.append(record)
            timings.append(proc_time)
            for metric_name in PAPER_METRICS:
                metrics[metric_name].append(record[metric_name])

    mean_time = float(np.mean(np.asarray(timings, dtype=np.float64)))
    fps = 1.0 / mean_time if mean_time > 0 else 0.0
    efficiency_info = OrderedDict(
        [
            ("Params_M", f"{params_m:.6f}"),
            ("Mean_Time_s", f"{mean_time:.6f}"),
            ("FPS", f"{fps:.6f}"),
        ]
    )
    per_image_path, summary_path, efficiency_path = write_metric_reports(test_out_folder, records, metrics, efficiency_info)

    print("\n" + "=" * 72)
    print("CT-MRI fusion metrics (mean +- std)")
    print("-" * 72)
    for metric_name in PAPER_METRICS:
        values = np.asarray(metrics[metric_name], dtype=np.float64)
        print(f"{metric_name:<9}: {np.mean(values):.4f} +- {np.std(values):.4f}")
    print("-" * 72)
    print(f"Time      : {mean_time:.4f}s")
    print(f"Params_M  : {params_m:.4f}")
    print(f"FPS       : {fps:.4f}")
    print("-" * 72)
    print(f"Evaluated images: {len(records)}")
    print(f"Per-image CSV  : {per_image_path}")
    print(f"Summary CSV    : {summary_path}")
    print(f"Efficiency CSV : {efficiency_path}")
    print("=" * 72)
