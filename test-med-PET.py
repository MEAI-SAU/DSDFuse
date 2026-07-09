import argparse
import csv
import os
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torch.nn.parameter import UninitializedParameter

import model as Model
from util.img_read_save import img_save
from util.util import RGB2YCrCb, YCrCb2RGB

try:
    from sewar.full_ref import vifp
except Exception as exc:
    vifp = None
    VIF_IMPORT_ERROR = exc
else:
    VIF_IMPORT_ERROR = None


EPS = 1e-12
PAPER_METRICS = ("MI", "VIF", "QABF", "SSIM", "NCIE", "FMI_pixel")


def count_initialized_parameters(net):
    return sum(
        param.nelement()
        for param in net.parameters()
        if not isinstance(param, UninitializedParameter)
    )


def tensor_to_uint8(image_tensor, value_range):
    low, high = value_range
    image = image_tensor[:1].detach().float().cpu().numpy().squeeze(0)
    if image.ndim == 3:
        image = np.transpose(image, (1, 2, 0))
    image = (image - low) / (high - low)
    image = np.clip(image, 0.0, 1.0)
    return np.rint(image * 255.0).astype(np.uint8)


def tensor_to_gray_u8(image_tensor, value_range):
    image = tensor_to_uint8(image_tensor, value_range)
    if image.ndim == 3:
        if image.shape[2] == 1:
            return image[:, :, 0]
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def tensor_to_rgb_u8(image_tensor, value_range):
    image = tensor_to_uint8(image_tensor, value_range)
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    return image


def ensure_gray_u8(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.clip(image, 0, 255).astype(np.uint8)


def calculate_q_en(image):
    image = ensure_gray_u8(image)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    prob = hist / (hist.sum() + EPS)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob + EPS)))


def calculate_q_sd(image):
    image = ensure_gray_u8(image).astype(np.float64)
    return float(np.std(image))


def calculate_ag(image):
    image = ensure_gray_u8(image).astype(np.float64)
    gradient_x = sobel(image, axis=1, mode="reflect")
    gradient_y = sobel(image, axis=0, mode="reflect")
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    return float(np.mean(gradient_magnitude))


def calculate_q_sf(image):
    image = ensure_gray_u8(image).astype(np.float64)
    row_freq = np.sqrt(np.mean(np.diff(image, axis=1) ** 2))
    col_freq = np.sqrt(np.mean(np.diff(image, axis=0) ** 2))
    return float(np.sqrt(row_freq ** 2 + col_freq ** 2))


def mutual_information_u8(a, b, bins=256):
    a = ensure_gray_u8(a)
    b = ensure_gray_u8(b)
    hist_2d, _, _ = np.histogram2d(
        a.ravel(),
        b.ravel(),
        bins=bins,
        range=[[0, 256], [0, 256]],
    )
    pxy = hist_2d / (hist_2d.sum() + EPS)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    px_py = np.outer(px, py)
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log2((pxy[nz] + EPS) / (px_py[nz] + EPS))))


def calculate_q_mi(x, y, f):
    x = ensure_gray_u8(x)
    y = ensure_gray_u8(y)
    f = ensure_gray_u8(f)
    return float(mutual_information_u8(x, f) + mutual_information_u8(y, f))


def entropy_u8(a, bins=256):
    a = ensure_gray_u8(a)
    hist = np.histogram(a.ravel(), bins=bins, range=(0, 256))[0].astype(np.float64)
    prob = hist / (hist.sum() + EPS)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob + EPS)))


def normalized_mutual_information_u8(a, b):
    mi = mutual_information_u8(a, b)
    ha = entropy_u8(a)
    hb = entropy_u8(b)
    return float((2.0 * mi) / (ha + hb + EPS))


def calculate_fmi_pixel(x, y, f):
    x = ensure_gray_u8(x)
    y = ensure_gray_u8(y)
    f = ensure_gray_u8(f)
    return float((normalized_mutual_information_u8(f, x) + normalized_mutual_information_u8(f, y)) / 2.0)


def entropy_base_b_from_hist(hist, base=256):
    hist = np.asarray(hist, dtype=np.float64)
    prob = hist / (hist.sum() + EPS)
    prob = prob[prob > 0]
    return float(-np.sum(prob * (np.log(prob + EPS) / np.log(base))))


def joint_entropy_base_b(a, b, bins=256):
    a = ensure_gray_u8(a)
    b = ensure_gray_u8(b)
    hist_2d, _, _ = np.histogram2d(
        a.ravel(),
        b.ravel(),
        bins=bins,
        range=[[0, 256], [0, 256]],
    )
    return entropy_base_b_from_hist(hist_2d, base=bins)


def nonlinear_correlation_coefficient(a, b, bins=256):
    a = ensure_gray_u8(a)
    b = ensure_gray_u8(b)
    hist_a = np.histogram(a.ravel(), bins=bins, range=(0, 256))[0]
    hist_b = np.histogram(b.ravel(), bins=bins, range=(0, 256))[0]
    h_a = entropy_base_b_from_hist(hist_a, base=bins)
    h_b = entropy_base_b_from_hist(hist_b, base=bins)
    h_ab = joint_entropy_base_b(a, b, bins=bins)
    mi_nat = (h_a + h_b - h_ab) * np.log(float(bins))
    return float(np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * mi_nat))))


def calculate_ncie(x, y, f):
    x = ensure_gray_u8(x)
    y = ensure_gray_u8(y)
    f = ensure_gray_u8(f)

    corr_matrix = np.array(
        [
            [1.0, nonlinear_correlation_coefficient(x, y), nonlinear_correlation_coefficient(x, f)],
            [nonlinear_correlation_coefficient(y, x), 1.0, nonlinear_correlation_coefficient(y, f)],
            [nonlinear_correlation_coefficient(f, x), nonlinear_correlation_coefficient(f, y), 1.0],
        ],
        dtype=np.float64,
    )
    eigenvalues = np.real(np.linalg.eigvals(corr_matrix))
    eigenvalues = np.clip(eigenvalues, EPS, None)
    normalized_eigenvalues = eigenvalues / 3.0
    return float(
        1.0 + np.sum(
            normalized_eigenvalues
            * (np.log(normalized_eigenvalues + EPS) / np.log(256.0))
        )
    )


def corr2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom < EPS:
        return 0.0
    return float(np.sum(a * b) / denom)


def calculate_q_scd(x, y, f):
    x = ensure_gray_u8(x).astype(np.float64)
    y = ensure_gray_u8(y).astype(np.float64)
    f = ensure_gray_u8(f).astype(np.float64)
    return float(corr2(f - x, y) + corr2(f - y, x))


def gradient_strength_orientation(image):
    image = ensure_gray_u8(image).astype(np.float64)
    grad_x = sobel(image, axis=1, mode="reflect")
    grad_y = sobel(image, axis=0, mode="reflect")
    grad = np.hypot(grad_x, grad_y)
    angle = np.mod(np.arctan2(grad_y, grad_x + EPS), np.pi)
    return grad, angle


def angle_similarity(source_angle, fused_angle):
    angle_diff = np.abs(source_angle - fused_angle)
    angle_diff = np.minimum(angle_diff, np.pi - angle_diff)
    similarity = 1.0 - angle_diff / (np.pi / 2.0)
    return np.clip(similarity, 0.0, 1.0)


def calculate_qabf(x, y, f):
    x = ensure_gray_u8(x)
    y = ensure_gray_u8(y)
    f = ensure_gray_u8(f)

    grad_x, angle_x = gradient_strength_orientation(x)
    grad_y, angle_y = gradient_strength_orientation(y)
    grad_f, angle_f = gradient_strength_orientation(f)

    tg = 0.9994
    kg = -15.0
    dg = 0.5
    ta = 0.9879
    ka = -22.0
    da = 0.8

    grad_ratio_x = np.where(grad_x > grad_f, grad_f / (grad_x + EPS), grad_x / (grad_f + EPS))
    grad_ratio_y = np.where(grad_y > grad_f, grad_f / (grad_y + EPS), grad_y / (grad_f + EPS))
    angle_ratio_x = angle_similarity(angle_x, angle_f)
    angle_ratio_y = angle_similarity(angle_y, angle_f)

    qg_x = tg / (1.0 + np.exp(kg * (grad_ratio_x - dg)))
    qg_y = tg / (1.0 + np.exp(kg * (grad_ratio_y - dg)))
    qa_x = ta / (1.0 + np.exp(ka * (angle_ratio_x - da)))
    qa_y = ta / (1.0 + np.exp(ka * (angle_ratio_y - da)))

    q_x = qg_x * qa_x
    q_y = qg_y * qa_y

    denominator = np.sum(grad_x + grad_y)
    if denominator < EPS:
        return 0.0

    numerator = np.sum(q_x * grad_x + q_y * grad_y)
    return float(numerator / (denominator + EPS))


def ssim_window_size(image):
    min_side = min(image.shape[0], image.shape[1])
    window = min(11, min_side)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        raise ValueError("SSIM requires image dimensions of at least 3x3.")
    return window


def calculate_q_ssim(x, y, f):
    x = ensure_gray_u8(x)
    y = ensure_gray_u8(y)
    f = ensure_gray_u8(f)
    win_size = ssim_window_size(f)
    ssim_x = float(ssim(x, f, win_size=win_size, data_range=255))
    ssim_y = float(ssim(y, f, win_size=win_size, data_range=255))
    return float((ssim_x + ssim_y) / 2.0), ssim_x, ssim_y


def sanitize_metric(value, default=0.0):
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return float(default)
    return value


def calculate_q_vif(x, y, f):
    if vifp is None:
        raise ImportError(
            f"sewar is required for standard Q_VIF evaluation. Original import error: {VIF_IMPORT_ERROR}"
        )

    x = ensure_gray_u8(x).astype(np.float64)
    y = ensure_gray_u8(y).astype(np.float64)
    f = ensure_gray_u8(f).astype(np.float64)

    vif_x = sanitize_metric(vifp(x, f))
    vif_y = sanitize_metric(vifp(y, f))
    return float((vif_x + vif_y) / 2.0), vif_x, vif_y


class InferenceFlopsWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net.test_Fusion(x, x.device)


def estimate_gflops(net, input_shape, device):
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception as exc:
        return None, f"torch.profiler unavailable: {exc}"

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    wrapper = InferenceFlopsWrapper(net).to(device)
    dummy_input = torch.randn(*input_shape, device=device)

    try:
        with torch.no_grad():
            _ = wrapper(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            with profile(activities=activities, with_flops=True) as prof:
                _ = wrapper(dummy_input)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

        total_flops = prof.key_averages().total_average().flops
        if total_flops is None or total_flops <= 0:
            return None, "Profiler returned zero FLOPs."

        return float(total_flops / 1e9), None
    except Exception as exc:
        return None, str(exc)


def write_metric_reports(output_folder, records, metrics, efficiency_info):
    per_image_path = os.path.join(output_folder, "metrics_per_image.csv")
    summary_path = os.path.join(output_folder, "metrics_summary.csv")
    efficiency_path = os.path.join(output_folder, "metrics_efficiency.csv")

    per_image_fields = [
        "image",
        "MI",
        "VIF",
        "QABF",
        "SSIM",
        "NCIE",
        "FMI_pixel",
        "SCD",
    ]

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
            if values.size == 0:
                writer.writerow([key, "", ""])
                continue
            writer.writerow([key, f"{np.mean(values):.6f}", f"{np.std(values):.6f}"])

    with open(efficiency_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["item", "value"])
        for key, value in efficiency_info.items():
            writer.writerow([key, value])

    return per_image_path, summary_path, efficiency_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/test_pet_mri.json",
        help="JSON file for configuration",
    )
    parser.add_argument("-local_rank", "--local_rank", type=int, default=0)
    parser.add_argument("-gpu", "--gpu_ids", type=str, default=None)
    parser.add_argument(
        "--resume_state",
        type=str,
        default=None,
        help="Override the checkpoint path in the config.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model checkpoint (overrides resume_state)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for fused images",
    )
    parser.add_argument(
        "--save_metrics",
        type=str,
        default=None,
        help="Path to save metrics CSV file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="PET-MRI",
        help="Dataset name: PET-MRI, SPECT-MRI or CT-MRI",
    )
    parser.add_argument(
        "--profile_flops",
        action="store_true",
        help="Estimate inference GFLOPs once using the first sample shape. This may be slow.",
    )
    parser.add_argument("--max_images", type=int, default=None, help="Evaluate only the first N images for smoke tests.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    with open(args.config, "r", encoding="utf-8-sig") as file:
        json_str = ""
        for line in file:
            json_str += line.split("//")[0] + "\n"
        opt = __import__("json").loads(json_str, object_pairs_hook=OrderedDict)
    
    # Ensure path section exists
    if "path" not in opt:
        opt["path"] = OrderedDict()
    if "resume_state" not in opt["path"]:
        opt["path"]["resume_state"] = None

    # Use --model_path if provided
    if args.model_path is not None:
        opt["path"]["resume_state"] = args.model_path
    elif args.resume_state is not None:
        opt["path"]["resume_state"] = args.resume_state

    if args.gpu_ids is not None:
        gpu_list = args.gpu_ids
    else:
        gpu_list = ",".join(str(x) for x in opt["gpu_ids"])

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    print("export CUDA_VISIBLE_DEVICES=" + gpu_list)

    opt["distributed"] = len(gpu_list.split(",")) > 1

    def dict_to_nonedict(value):
        if isinstance(value, dict):
            converted = {key: dict_to_nonedict(sub_value) for key, sub_value in value.items()}
            return type("NoneDict", (dict,), {"__missing__": lambda self, key: None})(**converted)
        if isinstance(value, list):
            return [dict_to_nonedict(sub_value) for sub_value in value]
        return value

    opt = dict_to_nonedict(opt)

    if vifp is None:
        raise ImportError(
            f"sewar is required for standard Q_VIF evaluation. Original import error: {VIF_IMPORT_ERROR}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this medical fusion inference pipeline because RGB/YCrCb reconstruction uses CUDA tensors."
        )

    value_range = (-1, 1) if opt["datasets"]["centered"] else (0, 1)

    # Use dataset from command line argument
    dataset_name = args.dataset
    print("The test result of " + dataset_name + " :")
    test_folder = os.path.join(script_dir, "dataset", "test", "Med", dataset_name)
    
    # Use output_dir from command line argument if provided
    if args.output_dir is not None:
        test_out_folder = args.output_dir
    else:
        test_out_folder = os.path.join(script_dir, "dataset", "test_result", "MED", dataset_name, opt["name"])

    metrics = OrderedDict(
        (key, [])
        for key in PAPER_METRICS
    )
    records = []
    timings = []
    failed_images = []

    if not os.path.exists(test_out_folder):
        os.makedirs(test_out_folder)

    device = torch.device("cuda")
    diffusion = Model.create_model(opt, args.local_rank)
    diffusion.Fusion_net.to(device)
    diffusion.Fusion_net.eval()

    total = count_initialized_parameters(diffusion.Fusion_net)
    params_m = total / 1e6
    print("Number of parameters: %.2fM" % params_m)

    gflops = None
    gflops_note = "Not profiled. Re-run with --profile_flops to estimate inference GFLOPs."
    gflops_input_shape = ""

    image_names = sorted(os.listdir(os.path.join(test_folder, "MRI")))
    if args.max_images is not None:
        image_names = image_names[: max(int(args.max_images), 0)]

    with torch.no_grad():
        for img_name in tqdm(image_names, desc="Processing images"):
            start_time = time.time()
            try:
                if dataset_name == "PET-MRI":
                    pet_folder = "PET"
                elif dataset_name == "SPECT-MRI":
                    pet_folder = "SPECT"
                else:
                    pet_folder = "CT"
                pet_pil = Image.open(os.path.join(test_folder, pet_folder, img_name)).convert("RGB")
                mri_pil = Image.open(os.path.join(test_folder, "MRI", img_name)).convert("L")

                pet_rgb = np.array(pet_pil, dtype=np.uint8)
                mri_gray = np.array(mri_pil, dtype=np.uint8)
                pet_gray = cv2.cvtColor(pet_rgb, cv2.COLOR_RGB2GRAY)

                pet_tensor = (
                    ToTensor()(pet_pil) * (value_range[1] - value_range[0]) + value_range[0]
                ).unsqueeze(0).to(device)
                mri_tensor = (
                    ToTensor()(mri_pil) * (value_range[1] - value_range[0]) + value_range[0]
                ).unsqueeze(0).to(device)

                pet_yuv = RGB2YCrCb(pet_tensor)
                pet_y = pet_yuv[:, 0:1, :, :]
                mri_y = mri_tensor[:, 0:1, :, :]
                input_tensor = torch.cat([mri_y, pet_y], dim=1)

                if args.profile_flops and gflops is None and not gflops_input_shape:
                    print(f"Profiling inference GFLOPs with input shape {tuple(input_tensor.shape)} ...")
                    gflops_input_shape = str(tuple(input_tensor.shape))
                    gflops, error = estimate_gflops(diffusion.Fusion_net, tuple(input_tensor.shape), device)
                    if error is not None:
                        gflops_note = f"GFLOPs estimation failed: {error}"
                        print(gflops_note)
                    else:
                        gflops_note = "Estimated with torch.profiler on the first sample shape."
                        print(f"Estimated GFLOPs: {gflops:.3f}")

                fused_y_tensor = diffusion.Fusion_net.test_Fusion(input_tensor, device)
                fused_rgb_tensor = YCrCb2RGB(
                    torch.cat((fused_y_tensor, pet_yuv[:, 1:2, :, :], pet_yuv[:, 2:3, :, :]), dim=1)
                )

                fused_rgb = tensor_to_rgb_u8(fused_rgb_tensor, value_range)
                fused_gray = tensor_to_gray_u8(fused_y_tensor, value_range)

                img_save(fused_rgb, img_name.split(".")[0], test_out_folder)

                target_h, target_w = fused_gray.shape[:2]
                pet_eval = cv2.resize(pet_gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                mri_eval = cv2.resize(mri_gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                fused_eval = fused_gray

                try:
                    q_vif, vif_pet, vif_mri = calculate_q_vif(pet_eval, mri_eval, fused_eval)
                except Exception as e:
                    q_vif = float('nan')
                    vif_pet = float('nan')
                    vif_mri = float('nan')
                    print(f"  VIF calculation skipped: {e}")
                q_ssim, ssim_pet, ssim_mri = calculate_q_ssim(pet_eval, mri_eval, fused_eval)
                q_mi = calculate_q_mi(pet_eval, mri_eval, fused_eval)
                q_abf = calculate_qabf(pet_eval, mri_eval, fused_eval)
                q_ncie = calculate_ncie(pet_eval, mri_eval, fused_eval)
                fmi_pixel = calculate_fmi_pixel(pet_eval, mri_eval, fused_eval)
                q_scd = calculate_q_scd(pet_eval, mri_eval, fused_eval)
                proc_time = time.time() - start_time

                record = OrderedDict(
                    [
                        ("image", img_name),
                        ("MI", float(q_mi)),
                        ("VIF", float(q_vif)),
                        ("QABF", float(q_abf)),
                        ("SSIM", float(q_ssim)),
                        ("NCIE", float(q_ncie)),
                        ("FMI_pixel", float(fmi_pixel)),
                        ("SCD", float(q_scd)),
                    ]
                )

                records.append(record)
                timings.append(float(proc_time))
                for metric_name in PAPER_METRICS:
                    metrics[metric_name].append(record[metric_name])

                print(f"\nImage: {img_name}")
                print(f"  MI       : {q_mi:.4f}")
                print(f"  VIF      : {q_vif:.4f}")
                print(f"  QABF     : {q_abf:.4f}")
                print(f"  SSIM     : {q_ssim:.4f}")
                print(f"  NCIE     : {q_ncie:.4f}")
                print(f"  FMI_pixel: {fmi_pixel:.4f}")
                print(f"  SCD      : {q_scd:.4f}")
                print(f"  Time     : {proc_time:.4f}s")

            except Exception as exc:
                failed_images.append((img_name, str(exc)))
                print(f"\nError calculating metrics for {img_name}: {exc}")

    if not records:
        raise RuntimeError("No PET-MRI images were evaluated successfully.")

    mean_time = float(np.mean(np.asarray(timings, dtype=np.float64)))
    fps = 1.0 / mean_time if mean_time > 0 else 0.0
    efficiency_info = OrderedDict(
        [
            ("Params_M", f"{params_m:.6f}"),
            ("Mean_Time_s", f"{mean_time:.6f}"),
            ("FPS", f"{fps:.6f}"),
            ("GFLOPs", "" if gflops is None else f"{gflops:.6f}"),
            ("GFLOPs_InputShape", gflops_input_shape),
            ("GFLOPs_Note", gflops_note),
        ]
    )

    per_image_path, summary_path, efficiency_path = write_metric_reports(
        test_out_folder, records, metrics, efficiency_info
    )

    # Save ablation-style metrics.csv if requested
    if args.save_metrics is not None:
        ablation_metrics_path = args.save_metrics
        os.makedirs(os.path.dirname(ablation_metrics_path), exist_ok=True)

        # Calculate mean values for the 7 required metrics
        mi_mean = float(np.mean(np.asarray(metrics["MI"], dtype=np.float64)))
        vif_mean = float(np.mean(np.asarray(metrics["VIF"], dtype=np.float64)))
        qabf_mean = float(np.mean(np.asarray(metrics["QABF"], dtype=np.float64)))
        ssim_mean = float(np.mean(np.asarray(metrics["SSIM"], dtype=np.float64)))
        ncie_mean = float(np.mean(np.asarray(metrics["NCIE"], dtype=np.float64)))
        fmi_mean = float(np.mean(np.asarray(metrics["FMI_pixel"], dtype=np.float64)))

        # Calculate SCD
        scd_values = np.asarray([record.get("SCD", np.nan) for record in records], dtype=np.float64)
        scd_values = scd_values[~np.isnan(scd_values)]
        scd_mean = float(np.mean(scd_values)) if scd_values.size else float("nan")

        file_exists = os.path.isfile(ablation_metrics_path)
        with open(ablation_metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Variant", "MI", "VIF", "Qabf", "SSIM", "NCIE", "FMI", "SCD"])
            # Extract variant name from model path or use default
            variant = "Unknown"
            if args.model_path is not None:
                variant = os.path.basename(os.path.dirname(os.path.dirname(args.model_path)))
            writer.writerow([variant, f"{mi_mean:.4f}", f"{vif_mean:.4f}", f"{qabf_mean:.4f}", f"{ssim_mean:.4f}", f"{ncie_mean:.4f}", f"{fmi_mean:.4f}", f"{scd_mean:.4f}"])
        print(f"Ablation metrics saved to: {ablation_metrics_path}")

    print("\n" + "=" * 72)
    print(f"{dataset_name} fusion metrics (mean +- std)")
    print("-" * 72)
    for metric_name in PAPER_METRICS:
        values = np.asarray(metrics[metric_name], dtype=np.float64)
        print(f"{metric_name:<8}: {np.mean(values):.4f} +- {np.std(values):.4f}")
    print("-" * 72)
    print(f"Time     : {mean_time:.4f}s")
    print(f"Params_M : {params_m:.4f}")
    print(f"FPS      : {fps:.4f}")
    if gflops is not None:
        print(f"GFLOPs   : {gflops:.4f}")
    else:
        print(f"GFLOPs   : N/A ({gflops_note})")
    print("-" * 72)
    print(f"Evaluated images: {len(records)}/{len(image_names)}")
    if failed_images:
        print(f"Failed images  : {len(failed_images)}")
    print(f"Per-image CSV  : {per_image_path}")
    print(f"Summary CSV    : {summary_path}")
    print(f"Efficiency CSV : {efficiency_path}")
    print("=" * 72)
