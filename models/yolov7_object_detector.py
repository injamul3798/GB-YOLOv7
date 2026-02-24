import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import xywh2xyxy, scale_coords
from utils.datasets import letterbox
import cv2
import time
import torchvision
import torch.nn as nn
from utils.general import box_iou


class YOLOV7TorchObjectDetector(nn.Module):
    def __init__(self,
                 model_weight,
                 device,
                 img_size,
                 names=None,
                 mode='eval',
                 confidence=0.45,
                 iou_thresh=0.45,
                 agnostic_nms=False):
        super(YOLOV7TorchObjectDetector, self).__init__()
        self.device = device
        self.img_size = img_size
        self.mode = mode
        self.confidence = confidence
        self.iou_thresh = iou_thresh
        self.agnostic = agnostic_nms

        # -------- LOAD MODEL --------
        self.model = attempt_load(model_weight, map_location=device)
        print("[INFO] Model is loaded")
        self.model.requires_grad_(True)
        self.model.to(device)
        self.model.eval()

        # -------- CLASS NAMES --------
        if names is None:
            self.names = ['trashcan', 'slippers', 'wire', 'socks']
        else:
            self.names = names

        # -------- WARMUP --------
        dummy = torch.zeros((1, 3, *self.img_size), device=device)
        self.model(dummy)

    # ============================================================
    # NON-MAX SUPPRESSION (ROBUST TO LOGITS LIST)
    # ============================================================
    @staticmethod
    def non_max_suppression(prediction, logits, conf_thres=0.3, iou_thres=0.45,
                            classes=None, agnostic=False, max_det=300):

        # ---- FIX: some GB-YOLOv7 forks return logits as list ----
        if isinstance(logits, list):
            b = prediction.shape[0]
            nc = prediction.shape[2] - 5
            logits = torch.zeros(
                (b, prediction.shape[1], nc),
                device=prediction.device
            )

        nc = prediction.shape[2] - 5
        xc = prediction[..., 4] > conf_thres

        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        logits_output = [torch.zeros((0, nc), device=prediction.device)] * prediction.shape[0]

        for xi, (x, log_) in enumerate(zip(prediction, logits)):
            x = x[xc[xi]]
            log_ = log_[xc[xi]]

            if not x.shape[0]:
                continue

            # Avoid in-place ops here to keep autograd graph valid for Grad-CAM backward.
            x = x.clone()
            scores = x[:, 5:] * x[:, 4:5]
            box = xywh2xyxy(x[:, :4])

            conf, j = scores.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            log_ = log_[conf.view(-1) > conf_thres]

            if not x.shape[0]:
                continue

            boxes, scores = x[:, :4], x[:, 4]
            i = torchvision.ops.nms(boxes, scores, iou_thres)
            i = i[:max_det]

            output[xi] = x[i]
            logits_output[xi] = log_[i]

        return output, logits_output

    # ============================================================
    # FORWARD (ROBUST TO 2 / 3 OUTPUT YOLOV7)
    # ============================================================
    def forward(self, img):
        out = self.model(img, augment=False)

        # ---- FIX: handle 2-output vs 3-output forks ----
        if isinstance(out, (list, tuple)) and len(out) == 3:
            prediction, logits, _ = out
        elif isinstance(out, (list, tuple)) and len(out) == 2:
            prediction, logits = out
        else:
            raise RuntimeError(
                f"Unexpected YOLOv7 output format: {type(out)}"
            )

        prediction, logits = self.non_max_suppression(
            prediction, logits,
            self.confidence,
            self.iou_thresh,
            agnostic=self.agnostic
        )

        self.boxes, self.classes, self.class_names, self.confidences = \
            [[[] for _ in range(img.shape[0])] for _ in range(4)]

        for i, det in enumerate(prediction):
            if len(det):
                scaled_xyxy = scale_coords(
                    img.shape[2:],
                    det[:, :4].clone(),
                    self._orig_shapes[i],
                    ratio_pad=self._ratio_pads[i],
                ).round()

                for row_idx, (*_, conf, cls) in enumerate(det):
                    bbox = [int(b) for b in scaled_xyxy[row_idx]]
                    cls = int(cls.item())

                    self.boxes[i].append(bbox)
                    self.confidences[i].append(conf)
                    self.classes[i].append(cls)
                    self.class_names[i].append(self.names[cls])

        return [self.boxes, self.classes, self.class_names, self.confidences], logits

    # ============================================================
    # PREPROCESSING
    # ============================================================
    def preprocessing(self, img):
        if len(img.shape) != 4:
            img = np.expand_dims(img, axis=0)

        letterboxed = []
        self._orig_shapes = []
        self._ratio_pads = []
        for im in img:
            self._orig_shapes.append(im.shape[:2])
            lb_img, ratio, pad = letterbox(im, new_shape=self.img_size)
            self._ratio_pads.append((ratio, pad))
            letterboxed.append(lb_img)
        img = np.array(letterboxed)

        img = img.transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float() / 255.0
        return img


if __name__ == '__main__':
    model_path = '../weights/yolov7_100e_64b_pre.pt'
    img_path = '../data/odsrihs/000001.jpg'
    model = YOLOV7TorchObjectDetector(model_path, 'cpu', img_size=(640, 640)).to('cpu')
    img = np.expand_dims(cv2.imread(img_path)[..., ::-1], axis=0)
    img = model.preprocessing(img)
    a = model(img)
    print(model._modules)
