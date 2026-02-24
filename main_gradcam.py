import os
import argparse
import numpy as np
import cv2
import torch

from models.gradcam import YOLOV7GradCAM, YOLOV7GradCAMPP
from models.yolov7_object_detector import YOLOV7TorchObjectDetector

# ================= CLASS NAMES ====================
names = ["nml", "benign", "malignant"]

# Use valid layer from your model printed modules
target_layers = ["96.act"] 

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, required=True,
                    help='Path to your trained GB-YOLOv7 weights (.pt)')
parser.add_argument('--img-path', type=str, required=True,
                    help='Path to input ultrasound image')
parser.add_argument('--output-dir', type=str, default='outputs/',
                    help='Where to save Grad-CAM outputs')
parser.add_argument('--img-size', type=int, default=640,
                    help="Image size to use")
parser.add_argument('--method', type=str, default='gradcam', choices=['gradcam','gradcampp'],
                    help='Method for CAM: gradcam or gradcampp')
parser.add_argument('--device', type=str, default='cuda',
                    help='cuda or cpu')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


def disable_inplace_ops(model):
    """Disable inplace=True ops (needed for Grad-CAM backprop stability)."""
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False


def get_heatmap_and_overlay(mask, res_img):
    """
    mask: torch tensor [1, H, W] or [1,1,H,W] depending on your gradcam output
    res_img: original BGR image (H,W,3)
    """
    if mask.ndim == 4:
        mask = mask.squeeze(0)  # [1,H,W]
    if mask.ndim == 3:
        mask = mask.squeeze(0)  # [H,W]

    cam = mask.detach().cpu().numpy().astype(np.float32)
    cam = np.clip(cam, 0.0, 1.0)
    # Remove low-activation haze so background keeps original appearance.
    thr = 0.30
    cam = np.maximum(cam - thr, 0.0) / (1.0 - thr + 1e-6)
    heat = (cam * 255.0).astype(np.uint8)

    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    # CAM is produced at model/input resolution; resize to original image for overlay.
    if heatmap.shape[:2] != res_img.shape[:2]:
        cam = cv2.resize(
            cam,
            (res_img.shape[1], res_img.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        heatmap = cv2.resize(
            heatmap,
            (res_img.shape[1], res_img.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    heatmap = heatmap.astype(np.float32) / 255.0

    base_img = res_img.astype(np.float32) / 255.0
    alpha = cam[..., None] * 0.75
    overlay = base_img * (1.0 - alpha) + heatmap * alpha
    overlay = (overlay * 255).astype(np.uint8)
    return overlay


def draw_box(img, box, label=None, color=(0, 255, 0)):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        text_thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        y_text = max(th + baseline, y1 - 6)
        # Draw black outline first for readability over heatmaps.
        cv2.putText(img, label, (x1, y_text),
                    font, font_scale, (0, 0, 0), text_thickness + 2, cv2.LINE_AA)
        cv2.putText(img, label, (x1, y_text),
                    font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
    return img


def run_gradcam(img_path):
    print(f"\n[INFO] Running Grad-CAM on: {img_path}")
    torch.autograd.set_detect_anomaly(True)

    orig = cv2.imread(img_path)
    if orig is None:
        print("Image not found:", img_path)
        return

    # ---- load model
    detector = YOLOV7TorchObjectDetector(
        args.model_path,
        args.device,
        img_size=(args.img_size, args.img_size),
        names=names
    )

    # ---- CRITICAL: disable fused path + inplace ops for Grad-CAM
    try:
        detector.model.model[-1].fuse = False
    except Exception:
        pass

    detector.model.requires_grad_(True)
    disable_inplace_ops(detector.model)

    # ---- prepare input
    img_rgb = orig[..., ::-1]
    torch_input = detector.preprocessing(img_rgb)

    for target_layer in target_layers:
        if args.method == 'gradcam':
            saliency_method = YOLOV7GradCAM(
                model=detector,
                layer_name=target_layer,
                img_size=(args.img_size, args.img_size),
            )
        else:
            saliency_method = YOLOV7GradCAMPP(
                model=detector,
                layer_name=target_layer,
                img_size=(args.img_size, args.img_size),
            )

        masks, preds = saliency_method(torch_input)
        boxes, _, class_names, conf = preds

        if len(boxes[0]) == 0:
            print("[WARNING] No detections found; Grad-CAM cannot be produced for this image.")
            return

        for i, mask in enumerate(masks):
            bbox = boxes[0][i]
            cls = class_names[0][i]
            score = conf[0][i]

            # tensor -> float for printing
            score_val = float(score.detach().cpu()) if isinstance(score, torch.Tensor) else float(score)

            overlay = get_heatmap_and_overlay(mask, orig.copy())
            labeled = draw_box(overlay, bbox, f"{cls} {score_val:.2f}")

            base = os.path.basename(img_path).split('.')[0]
            out_name = f"{base}_{target_layer.replace('.', '_')}_{i}.jpg"
            out_path = os.path.join(args.output_dir, out_name)

            cv2.imwrite(out_path, labeled)
            print(f"[SAVED] {out_path}")

    print("[INFO] Finished Grad-CAM generation!")


if __name__ == "__main__":
    run_gradcam(args.img_path)
