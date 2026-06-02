"""
model.py — Brain Tumor ResNet-50 inference + Grad-CAM
"""

import io
import base64
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = "model/brain_tumor_resnet50.pth"
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
DEVICE = torch.device("cpu")

# ImageNet normalisation (same as training)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
_model = None


def _build_model():
    """Build ResNet-50 with 4-class head and load trained weights."""
    net = models.resnet50(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    net.load_state_dict(state_dict)
    net.to(DEVICE)
    net.eval()
    return net


def get_model():
    """Return the cached model (loaded once)."""
    global _model
    if _model is None:
        _model = _build_model()
    return _model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(image_bytes: bytes) -> tuple:
    """Return (input_tensor [1,3,224,224], original PIL image)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    return tensor, img


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(image_bytes: bytes) -> dict:
    """
    Run inference and return:
      class_name, confidence (%), per-class probabilities dict
    """
    model = get_model()
    tensor, _ = preprocess(image_bytes)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze()

    confidence, idx = probs.max(0)
    class_name = CLASS_NAMES[idx.item()]

    probabilities = {name: round(p.item() * 100, 2) for name, p in zip(CLASS_NAMES, probs)}

    return {
        "class": class_name,
        "confidence": round(confidence.item() * 100, 2),
        "probabilities": probabilities,
    }


# ---------------------------------------------------------------------------
# Grad-CAM++ with multi-layer fusion
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm


def _register_hooks_on_layer(layer):
    """Register forward/backward hooks on a given layer."""
    store = {"activations": None, "gradients": None}

    def fwd_hook(_module, _input, output):
        store["activations"] = output.detach()

    def bwd_hook(_module, _grad_in, grad_out):
        store["gradients"] = grad_out[0].detach()

    h_fwd = layer.register_forward_hook(fwd_hook)
    h_bwd = layer.register_full_backward_hook(bwd_hook)
    return store, h_fwd, h_bwd


def _compute_gradcam_pp(activations, gradients):
    """
    Grad-CAM++ weighting: uses second- and third-order gradients
    to produce better-localised heatmaps, especially for small regions.
    """
    grads = gradients             # [1, C, H, W]
    acts  = activations           # [1, C, H, W]

    # Second and third order
    grads_2 = grads ** 2
    grads_3 = grads ** 3

    # Alpha coefficients (Grad-CAM++ formula)
    denom = 2.0 * grads_2 + (grads_3 * acts).sum(dim=(2, 3), keepdim=True)
    denom = torch.where(denom != 0, denom, torch.ones_like(denom))
    alphas = grads_2 / denom

    # Weights: ReLU(grads) * alpha, then global sum over spatial dims
    weights = (alphas * F.relu(grads)).sum(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

    # Weighted combination of activation maps
    cam = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, H, W]
    cam = F.relu(cam)
    return cam.squeeze()  # [H, W]


def _normalize_cam(cam):
    """Normalize a CAM to [0, 1] with robust min-max."""
    cam = cam.numpy() if isinstance(cam, torch.Tensor) else cam
    if cam.max() > 0:
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
    return cam


def generate_gradcam(image_bytes: bytes) -> str:
    """
    Compute Grad-CAM++ with layer3+layer4 fusion.
    Returns a base64-encoded PNG of the heatmap overlaid on the original image.

    Improvements over vanilla Grad-CAM:
      - Grad-CAM++ alpha weighting for better small-region localisation
      - Multi-layer fusion: layer3 (14x14, fine detail) + layer4 (7x7, semantics)
      - BICUBIC upsampling for smoother heatmaps
      - Adaptive blending: heatmap is stronger where activation is high
    """
    model = get_model()
    tensor, original_img = preprocess(image_bytes)
    tensor.requires_grad_(True)

    # Hook into both layer3 and layer4
    store3, h3_fwd, h3_bwd = _register_hooks_on_layer(model.layer3)
    store4, h4_fwd, h4_bwd = _register_hooks_on_layer(model.layer4)

    try:
        # Forward pass
        logits = model(tensor)
        pred_idx = logits.argmax(dim=1).item()

        # Backward pass for predicted class
        model.zero_grad()
        logits[0, pred_idx].backward()

        # Compute Grad-CAM++ for each layer
        cam4 = _compute_gradcam_pp(store4["activations"], store4["gradients"])
        cam3 = _compute_gradcam_pp(store3["activations"], store3["gradients"])

        # Resize both CAMs to the target output size
        out_size = (224, 224)

        cam4_up = F.interpolate(
            cam4.unsqueeze(0).unsqueeze(0), size=out_size,
            mode="bicubic", align_corners=False
        ).squeeze().clamp(min=0)

        cam3_up = F.interpolate(
            cam3.unsqueeze(0).unsqueeze(0), size=out_size,
            mode="bicubic", align_corners=False
        ).squeeze().clamp(min=0)

        # Fuse: weighted combination — layer4 for semantics, layer3 for detail
        fused = 0.55 * _normalize_cam(cam4_up) + 0.45 * _normalize_cam(cam3_up)
        fused = _normalize_cam(fused)

        # Apply a mild power transform to increase contrast on hot spots
        fused = np.power(fused, 0.8)
        fused = _normalize_cam(fused)

        # Resize to original image dimensions
        cam_img = Image.fromarray((fused * 255).astype(np.uint8)).resize(
            original_img.size, Image.BICUBIC
        )
        cam_np = np.array(cam_img).astype(np.float64) / 255.0

        # Colourmap: use 'inferno' — better for medical imaging than 'jet'
        heatmap_rgba = cm.inferno(cam_np)  # [H, W, 4] float [0,1]
        heatmap_rgb = (heatmap_rgba[..., :3] * 255).astype(np.uint8)

        # Adaptive blending: where the heatmap is strong, show more heatmap
        original_np = np.array(
            original_img.resize(original_img.size)
        ).astype(np.float64)

        alpha = cam_np[..., np.newaxis]          # [H, W, 1] in [0, 1]
        blend_strength = 0.35 + 0.45 * alpha     # ranges from 0.35 to 0.80
        overlay = (
            (1.0 - blend_strength) * original_np +
            blend_strength * heatmap_rgb.astype(np.float64)
        )
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Encode to base64 PNG
        overlay_img = Image.fromarray(overlay)
        buf = io.BytesIO()
        overlay_img.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    finally:
        h3_fwd.remove()
        h3_bwd.remove()
        h4_fwd.remove()
        h4_bwd.remove()
