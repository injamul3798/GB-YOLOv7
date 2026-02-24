import time
import torch
import torch.nn.functional as F


# ============================================================
# FIND TARGET LAYER
# ============================================================
def find_yolo_layer(model, layer_name):
    """
    model: YOLOV7TorchObjectDetector
    layer_name: e.g. '103.act'
    """
    parts = layer_name.split(".")
    net = model.model.model  # actual YOLOv7 backbone

    for part in parts:
        if part.isdigit():
            net = list(net._modules.values())[int(part)]
        elif part in net._modules:
            net = net._modules[part]
        else:
            raise KeyError(f"Layer '{layer_name}' not found at '{part}'")

    return net


# ============================================================
# GRAD-CAM (DETECTOR SAFE)
# ============================================================
class YOLOV7GradCAM:
    """
    Stable Grad-CAM for YOLOv7 detectors.
    Uses tensor hook instead of backward hooks.
    """

    def __init__(self, model, layer_name, img_size=(640, 640)):
        self.model = model
        self.gradients = None
        self.activations = None

        def forward_hook(module, input, output):
          # output might be tuple/list in some layers
          if isinstance(output, (tuple, list)):
              output = output[0]
          self.activations = output
          output.register_hook(self._save_gradient)

        target_layer = find_yolo_layer(self.model, layer_name)
        target_layer.register_forward_hook(forward_hook)

        device = "cuda" if next(self.model.model.parameters()).is_cuda else "cpu"
        self.model(torch.zeros(1, 3, *img_size, device=device))

        print("[INFO] saliency_map size:", self.activations.shape[2:])

    def _save_gradient(self, grad):
        self.gradients = grad

    def forward(self, input_img):
        saliency_maps = []
        b, c, h, w = input_img.size()

        tic = time.time()
        preds, _ = self.model(input_img)
        print("[INFO] model-forward took:", round(time.time() - tic, 4), "seconds")

        boxes, classes, class_names, confidences = preds

        # Grad-CAM only makes sense if at least one detection exists
        if len(confidences[0]) == 0:
            print("[WARNING] No detections found — skipping Grad-CAM.")
            return [], preds

        for det_idx in range(len(confidences[0])):

            score = confidences[0][det_idx]

            if not isinstance(score, torch.Tensor):
                score = torch.tensor(
                    score, device=input_img.device, requires_grad=True
                )

            self.model.zero_grad()
            tic = time.time()
            score.backward(retain_graph=True)
            print(
                f"[INFO] backward on detection {det_idx} took:",
                round(time.time() - tic, 4),
                "seconds",
            )

            if self.gradients is None:
                raise RuntimeError("Gradients not captured — hook failed.")

            gradients = self.gradients
            activations = self.activations

            b, k, u, v = gradients.size()

            # Global Average Pooling
            alpha = gradients.view(b, k, -1).mean(2)
            weights = alpha.view(b, k, 1, 1)

            saliency_map = (weights * activations).sum(1, keepdim=True)
            saliency_map = F.relu(saliency_map)

            saliency_map = F.interpolate(
                saliency_map, size=(h, w), mode="bilinear", align_corners=False
            )

            saliency_map -= saliency_map.min()
            saliency_map /= (saliency_map.max() + 1e-6)

            saliency_maps.append(saliency_map.detach())

        return saliency_maps, preds

    def __call__(self, input_img):
        return self.forward(input_img)


# ============================================================
# GRAD-CAM++
# ============================================================
class YOLOV7GradCAMPP(YOLOV7GradCAM):
    """
    Grad-CAM++ variant (same stable hook logic)
    """

    def forward(self, input_img):
        saliency_maps = []
        b, c, h, w = input_img.size()

        preds, _ = self.model(input_img)
        boxes, classes, class_names, confidences = preds

        if len(confidences[0]) == 0:
            print("[WARNING] No detections found — skipping Grad-CAM++.")
            return [], preds

        for det_idx in range(len(confidences[0])):

            score = confidences[0][det_idx]
            if not isinstance(score, torch.Tensor):
                score = torch.tensor(
                    score, device=input_img.device, requires_grad=True
                )

            self.model.zero_grad()
            score.backward(retain_graph=True)

            gradients = self.gradients
            activations = self.activations

            b, k, u, v = gradients.size()

            alpha_num = gradients.pow(2)
            alpha_denom = (
                gradients.pow(2) * 2
                + activations.mul(gradients.pow(3))
                .view(b, k, u * v)
                .sum(-1, keepdim=True)
                .view(b, k, 1, 1)
            )

            alpha_denom = torch.where(
                alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom)
            )

            alpha = alpha_num / (alpha_denom + 1e-7)
            positive_gradients = F.relu(score.exp() * gradients)

            weights = (
                (alpha * positive_gradients)
                .view(b, k, u * v)
                .sum(-1)
                .view(b, k, 1, 1)
            )

            saliency_map = (weights * activations).sum(1, keepdim=True)
            saliency_map = F.relu(saliency_map)

            saliency_map = F.interpolate(
                saliency_map, size=(h, w), mode="bilinear", align_corners=False
            )

            saliency_map -= saliency_map.min()
            saliency_map /= (saliency_map.max() + 1e-6)

            saliency_maps.append(saliency_map.detach())

        return saliency_maps, preds