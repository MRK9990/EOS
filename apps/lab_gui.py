import os
import json
import shutil

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import cv2  # OpenCV for LAB conversion
except Exception as e:  # pragma: no cover
    cv2 = None

import numpy as np
from PIL import Image, ImageTk

# Enable HEIC/HEIF reading if available
try:  # pragma: no cover
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass


def np_from_pil(img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to an RGB numpy array (uint8)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def pil_from_np(arr: np.ndarray) -> Image.Image:
    """Convert an RGB numpy array (uint8) to a PIL Image."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def rgb_to_lab_mean(rgb_roi):
    """
    Compute mean CIE L*, a*, b* for an RGB ROI.
    Returns (L, a, b) as floats where:
      L in [0..100], a in [-127..127], b in [-127..127]
    """
    import numpy as np
    import cv2

    if rgb_roi is None or getattr(rgb_roi, "size", 0) == 0:
        return (None, None, None)

    # rgb_roi is RGB (Pillow/Matplotlib convention). OpenCV expects BGR.
    bgr = rgb_roi[..., ::-1].astype(np.float32) / 255.0  # float32 in [0,1]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)           # now true CIE L*a*b*
    L = float(np.mean(lab[..., 0]))
    a = float(np.mean(lab[..., 1]))
    b = float(np.mean(lab[..., 2]))
    return (L, a, b)


def delta_e_cie76(lab1, lab2):
    """
    lab1, lab2: tuples/lists of (L*, a*, b*) in true CIE ranges
    Returns float ΔE76 = sqrt((dL)^2 + (da)^2 + (db)^2)
    """
    if not lab1 or not lab2:
        return None
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    return float(((L2 - L1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2) ** 0.5)


def delta_e_ciede2000(lab1, lab2, kL=1.0, kC=1.0, kH=1.0):
    """
    CIEDE2000 color-difference between two CIE L*a*b* colors.

    lab1, lab2: (L*, a*, b*) with L in [0..100], a,b approx [-127..127]
    Returns float ΔE00. See Sharma et al., 2005 (CIEDE2000).
    """
    if not lab1 or not lab2:
        return None

    L1, a1, b1 = map(float, lab1)
    L2, a2, b2 = map(float, lab2)

    # Step 1: Compute C' (prime) and h' (prime)
    avg_L = 0.5 * (L1 + L2)
    C1 = (a1 * a1 + b1 * b1) ** 0.5
    C2 = (a2 * a2 + b2 * b2) ** 0.5
    avg_C = 0.5 * (C1 + C2)

    G = 0.5 * (1 - (avg_C ** 7 / (avg_C ** 7 + 25 ** 7)) ** 0.5) if avg_C != 0 else 0.0

    a1p = (1 + G) * a1
    a2p = (1 + G) * a2

    C1p = (a1p * a1p + b1 * b1) ** 0.5
    C2p = (a2p * a2p + b2 * b2) ** 0.5

    import math

    def hp(a_prime, b):
        if a_prime == 0.0 and b == 0.0:
            return 0.0
        h = math.degrees(math.atan2(b, a_prime))
        return h + 360 if h < 0 else h

    h1p = hp(a1p, b1)
    h2p = hp(a2p, b2)

    # Step 2: ΔL', ΔC', ΔH'
    dLp = L2 - L1
    dCp = C2p - C1p

    def dhp(C1p_, C2p_, h1p_, h2p_):
        if C1p_ * C2p_ == 0.0:
            return 0.0
        dh = h2p_ - h1p_
        if dh > 180:
            dh -= 360
        elif dh < -180:
            dh += 360
        return dh

    dhp_val = dhp(C1p, C2p, h1p, h2p)
    dHp = 2.0 * (C1p * C2p) ** 0.5 * math.sin(math.radians(dhp_val / 2.0))

    # Step 3: Averages
    avg_Lp = 0.5 * (L1 + L2)
    avg_Cp = 0.5 * (C1p + C2p)

    def h_bar(C1p_, C2p_, h1p_, h2p_):
        if C1p_ * C2p_ == 0.0:
            return h1p_ + h2p_
        dh = abs(h1p_ - h2p_)
        if dh > 180:
            return (h1p_ + h2p_ + 360) * 0.5 if (h1p_ + h2p_) < 360 else (h1p_ + h2p_ - 360) * 0.5
        return 0.5 * (h1p_ + h2p_)

    avg_hp = h_bar(C1p, C2p, h1p, h2p)

    # Step 4: T term
    T = (
        1
        - 0.17 * math.cos(math.radians(avg_hp - 30))
        + 0.24 * math.cos(math.radians(2 * avg_hp))
        + 0.32 * math.cos(math.radians(3 * avg_hp + 6))
        - 0.20 * math.cos(math.radians(4 * avg_hp - 63))
    )

    # Step 5: SL, SC, SH
    Sl = 1 + (0.015 * (avg_Lp - 50) ** 2) / math.sqrt(20 + (avg_Lp - 50) ** 2)
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T

    # Step 6: RT
    delta_theta = 30 * math.exp(-((avg_hp - 275) / 25) ** 2)
    Rc = 2 * ((avg_Cp ** 7) / (avg_Cp ** 7 + 25 ** 7)) ** 0.5 if avg_Cp != 0 else 0.0
    Rt = -Rc * math.sin(math.radians(2 * delta_theta))

    # Step 7: ΔE00
    dE = (
        (dLp / (kL * Sl)) ** 2
        + (dCp / (kC * Sc)) ** 2
        + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    ) ** 0.5

    return float(dE)


def delta_components(lab_ref, lab_test):
    if not lab_ref or not lab_test:
        return None
    Lr, ar, br = lab_ref
    Lt, at, bt = lab_test
    return (Lt - Lr, at - ar, bt - br)


def pairwise_deltaE_and_components(labs_ref, labs_test):
    """
    labs_ref, labs_test: lists of length 3 with tuples (L,a,b) or (None,None,None)

    Returns:
      deltaEs       : list of length 3 with ΔE76 per ROI (or None)
      comps         : list of length 3 with (ΔL,Δa,Δb) per ROI (or None)
      overall_mean  : mean of available ΔE76 (or None)
      overall_pooled: ΔE76 from mean component differences
    """
    deltaEs, comps = [], []
    dLs, dAs, dBs = [], [], []

    for i in range(3):
        r = labs_ref[i] if i < len(labs_ref) else (None, None, None)
        t = labs_test[i] if i < len(labs_test) else (None, None, None)
        if r[0] is None or t[0] is None:
            deltaEs.append(None)
            comps.append(None)
            continue
        dL = t[0] - r[0]
        da = t[1] - r[1]
        db = t[2] - r[2]
        comps.append((dL, da, db))
        de = float((dL * dL + da * da + db * db) ** 0.5)
        deltaEs.append(de)
        dLs.append(dL)
        dAs.append(da)
        dBs.append(db)

    overall_mean = None
    vals = [v for v in deltaEs if v is not None]
    if vals:
        overall_mean = float(np.mean(vals))

    overall_pooled = None
    if dLs and dAs and dBs:
        dLbar = float(np.mean(dLs))
        dAbar = float(np.mean(dAs))
        dBbar = float(np.mean(dBs))
        overall_pooled = float((dLbar * dLbar + dAbar * dAbar + dBbar * dBbar) ** 0.5)

    return deltaEs, comps, overall_mean, overall_pooled


def pairwise_deltaEs(labs_ref, labs_test, metric="DE76"):
    """
    metric: "DE76" or "DE00"
    Returns per-roi ΔE list, and overall mean/pooled for that metric.
    Pooled uses component means as ΔE76 magnitude (see note in UI).
    """
    des, dLs, dAs, dBs = [], [], [], []
    for i in range(3):
        r = labs_ref[i] if i < len(labs_ref) else (None, None, None)
        t = labs_test[i] if i < len(labs_test) else (None, None, None)
        if r[0] is None or t[0] is None:
            des.append(None)
            continue
        if metric == "DE00":
            de = delta_e_ciede2000(r, t)
        else:
            de = delta_e_cie76(r, t)
        des.append(float(de))
        dLs.append(t[0] - r[0]); dAs.append(t[1] - r[1]); dBs.append(t[2] - r[2])

    mean_de = None
    vals = [v for v in des if v is not None]
    if vals:
        mean_de = float(np.mean(vals))

    pooled_de = None
    if dLs and dAs and dBs:
        dLbar = float(np.mean(dLs))
        dAbar = float(np.mean(dAs))
        dBbar = float(np.mean(dBs))
        pooled_de = float((dLbar * dLbar + dAbar * dAbar + dBbar * dBbar) ** 0.5)

    return des, mean_de, pooled_de


def apply_exposure_rgb(rgb, ev):
    """
    Adjust exposure in linear domain:
    1) convert sRGB->linear, 2) multiply by 2**ev, 3) back to sRGB.
    """
    import numpy as np
    if rgb is None:
        return None
    arr = rgb.astype(np.float32) / 255.0
    # sRGB -> linear
    thr = 0.04045
    lin = np.where(arr <= thr, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    # exposure
    lin *= (2.0 ** float(ev))
    # linear -> sRGB
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * (np.clip(lin, 0, 1) ** (1 / 2.4)) - 0.055)
    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)


def apply_adjustments(rgb: np.ndarray, r_gain: float, b_gain: float, brightness: float, contrast: float) -> np.ndarray:
    """Apply per-channel gains for white balance, then brightness/contrast.

    - r_gain: multiply red channel
    - b_gain: multiply blue channel
    - brightness: additive [-100..100] mapped directly to beta
    - contrast: scale factor around 1.0 in [0.5..1.5] from slider [-50..50]
    """
    if rgb is None:
        return None
    arr = rgb.astype(np.float32)
    # Correct mapping in RGB order: [R, G, B] -> indices [0, 1, 2]
    arr[..., 0] *= r_gain  # R
    # G unchanged
    arr[..., 2] *= b_gain  # B

    # Contrast (alpha) and brightness (beta)
    alpha = 1.0 + (contrast / 100.0)
    beta = brightness  # directly add in [−100..100]
    arr = arr * alpha + beta

    return np.clip(arr, 0, 255).astype(np.uint8)


class SampleManager:
    def __init__(self, root_dir="./dataset"):
        self.set_root(root_dir)

    def set_root(self, root_dir: str):
        base = root_dir or "./dataset"
        self.root_dir = os.path.abspath(base)
        self.samples_dir = os.path.join(self.root_dir, "samples")
        os.makedirs(self.samples_dir, exist_ok=True)

    def next_id(self) -> str:
        existing = []
        if os.path.isdir(self.samples_dir):
            for name in os.listdir(self.samples_dir):
                if len(name) == 6 and name.isdigit():
                    existing.append(int(name))
        nid = (max(existing) + 1) if existing else 1
        return f"{nid:06d}"

    def make_sample_folder(self):
        sid = self.next_id()
        path = os.path.join(self.samples_dir, sid)
        os.makedirs(path, exist_ok=True)
        return sid, path

    @staticmethod
    def _imwrite_np_or_pil(image_obj, out_path):
        if image_obj is None:
            raise ValueError("No image data to save")
        if isinstance(image_obj, Image.Image):
            image_obj.save(out_path)
            return
        if isinstance(image_obj, np.ndarray):
            arr = image_obj
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(out_path)
            return
        raise ValueError(f"Unsupported image type for saving: {type(image_obj)}")

    def save_sample(self, sample_path, ref_img, test_orig_img, test_adj_img, meta_dict):
        os.makedirs(sample_path, exist_ok=True)

        def _save(src, dst, fallback=None):
            if isinstance(src, str) and os.path.isfile(src):
                shutil.copy2(src, dst)
                return
            if src is None and fallback and os.path.isfile(fallback):
                shutil.copy2(fallback, dst)
                return
            if src is None:
                raise ValueError("Missing image source for saving")
            self._imwrite_np_or_pil(src, dst)

        # Reference image is not saved - only used temporarily for LAB comparison
        orig_path = os.path.join(sample_path, "test_original.jpg")
        adj_path = os.path.join(sample_path, "test_adjusted.jpg")

        _save(test_orig_img, orig_path)
        _save(test_adj_img, adj_path, fallback=test_orig_img if isinstance(test_orig_img, str) else None)

        meta_copy = dict(meta_dict)
        meta_copy["paths"] = {
            "test_original": "test_original.jpg",
            "test_adjusted": "test_adjusted.jpg",
        }
        with open(os.path.join(sample_path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_copy, f, ensure_ascii=False, indent=2)


class ROIState:
    def __init__(self):
        self.start = None  # (x, y) in canvas coords
        self.end = None    # (x, y) in canvas coords
        self.rect_id = None

    def set_start(self, x, y):
        self.start = (x, y)
        self.end = (x, y)

    def set_end(self, x, y):
        self.end = (x, y)

    def get_bbox(self):
        if not self.start or not self.end:
            return None
        x0, y0 = self.start
        x1, y1 = self.end
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


class ImagePanel:
    def __init__(self, parent, title: str, width=640, height=480):
        self.frame = ttk.Frame(parent)
        self.title_lbl = ttk.Label(self.frame, text=title)
        self.title_lbl.pack(anchor="w")
        self.canvas_w = width
        self.canvas_h = height
        self.canvas = tk.Canvas(self.frame, width=self.canvas_w, height=self.canvas_h, bg="#202020", highlightthickness=1, highlightbackground="#555")
        self.canvas.pack()
        # Enable keyboard focus and ROI slot hotkeys (1/2/3)
        self.canvas.configure(takefocus=True)
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())
        self.canvas.bind("<1>", lambda e: (self.canvas.focus_set(), None))
        self.canvas.bind("<Key-1>", lambda e: self.set_active_slot(0))
        self.canvas.bind("<Key-2>", lambda e: self.set_active_slot(1))
        self.canvas.bind("<Key-3>", lambda e: self.set_active_slot(2))
        # Zoom controls
        self._zoom_var = tk.DoubleVar(value=100.0)
        zoom_row = ttk.Frame(self.frame)
        zoom_row.pack(fill="x", pady=(4, 0))
        ttk.Label(zoom_row, text="Zoom", width=6).pack(side="left")
        self._zoom_scale = ttk.Scale(zoom_row, from_=25, to=400, orient="horizontal", variable=self._zoom_var,
                                     command=lambda v: self._on_zoom_slider())
        self._zoom_scale.pack(side="left", fill="x", expand=True, padx=6)
        self._zoom_lbl = ttk.Label(zoom_row, text="100%")
        self._zoom_lbl.pack(side="left")
        # Zoom buttons
        zoom_btns = ttk.Frame(self.frame)
        zoom_btns.pack(fill="x", padx=0, pady=(2, 0))
        ttk.Button(zoom_btns, text="-", width=3, command=lambda: self._zoom_button(False)).pack(side="left")
        ttk.Button(zoom_btns, text="Reset View", command=self.reset_view).pack(side="left", padx=6)
        ttk.Button(zoom_btns, text="+", width=3, command=lambda: self._zoom_button(True)).pack(side="left")

        # ROI slot selector
        slot_row = ttk.Frame(self.frame)
        slot_row.pack(fill="x", pady=(2, 0))
        ttk.Label(slot_row, text="ROI slot:", width=8).pack(side="left")
        def mk_btn(i, txt):
            return ttk.Button(slot_row, text=txt, width=6, command=lambda: self.set_active_slot(i))
        mk_btn(0, "ROI1").pack(side="left", padx=(4, 2))
        mk_btn(1, "ROI2").pack(side="left", padx=2)
        mk_btn(2, "ROI3").pack(side="left", padx=2)
        # Three LAB value labels (per ROI slot)
        self.lab_lbls = [
            ttk.Label(self.frame, text="ROI1  L*: -, a*: -, b*: -"),
            ttk.Label(self.frame, text="ROI2  L*: -, a*: -, b*: -"),
            ttk.Label(self.frame, text="ROI3  L*: -, a*: -, b*: -"),
        ]
        for lbl in self.lab_lbls:
            lbl.pack(anchor="w", pady=(2, 0))

        self.tk_img = None
        self.pil_img = None  # PIL Image in original resolution
        self.np_img = None   # RGB numpy array in original resolution

        # Drawing scale/mapping
        self.scale = 1.0        # actual scale (base_fit_scale * zoom)
        self.offset_x = 0       # full offset including pan
        self.offset_y = 0
        self._base_scale = 1.0  # fit-to-canvas scale
        self._base_off_x = 0
        self._base_off_y = 0
        self.zoom = 1.0         # zoom factor relative to fit (0.25..4.0)
        self.pan_x = 0          # pan relative to base offset
        self.pan_y = 0
        self._panning = False
        self._pan_start = None

        # ROIs (3 slots)
        self.rois = [ROIState(), ROIState(), ROIState()]
        self.active_slot = 0
        self.roi_colors = ["#00FF88", "#FFAA00", "#3399FF"]

        # ROI update callback (set by App)
        self.on_roi_changed = None

        # Bind events for ROI
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        # Mouse wheel zoom
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)        # Windows/Mac
        self.canvas.bind("<Button-4>", lambda e: self._on_mousewheel_linux(1))  # Linux scroll up
        self.canvas.bind("<Button-5>", lambda e: self._on_mousewheel_linux(-1)) # Linux scroll down
        # Pan with middle or right mouse button
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_end)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def load_image(self, pil_image: Image.Image):
        self.pil_img = pil_image
        self.np_img = np_from_pil(pil_image)
        self._redraw()

    def update_from_np(self, np_image: np.ndarray):
        self.np_img = np_image
        self.pil_img = pil_from_np(np_image)
        self._redraw()

    def clear(self):
        self.pil_img = None
        self.np_img = None
        self.tk_img = None
        self.canvas.delete("all")
        for i, lbl in enumerate(self.lab_lbls):
            lbl.config(text=f"ROI{i+1}  L*: -, a*: -, b*: -")
        self.rois = [ROIState(), ROIState(), ROIState()]

    def _compute_scale(self, img_w, img_h):
        """Compute draw dimensions from pure zoom factor.
        Scale = zoom (1.0 = 100% = 1:1 pixel mapping).
        No hidden fit-to-canvas base scaling.
        """
        if img_w == 0 or img_h == 0:
            return 0, 0
        draw_w = int(img_w * self.zoom)
        draw_h = int(img_h * self.zoom)
        return draw_w, draw_h

    def _redraw(self):
        self.canvas.delete("all")
        if self.pil_img is None:
            return
        img_w, img_h = self.pil_img.size
        draw_w, draw_h = self._compute_scale(img_w, img_h)
        # Clamp pan so image covers canvas appropriately
        self._clamp_pan(draw_w, draw_h)
        # Scale is pure zoom (1.0 = 100% = 1:1 pixel mapping)
        self.scale = self.zoom
        # Offset calculation: use pan directly when image exceeds canvas,
        # otherwise use centered offset to keep image centered.
        if draw_w > self.canvas_w:
            # Image wider than canvas: offset_x = pan_x
            self.offset_x = self.pan_x
        else:
            # Image fits canvas: center it
            self.offset_x = int((self.canvas_w - draw_w) / 2)
        
        if draw_h > self.canvas_h:
            # Image taller than canvas: offset_y = pan_y
            self.offset_y = self.pan_y
        else:
            # Image fits canvas: center it
            self.offset_y = int((self.canvas_h - draw_h) / 2)
        
        safe_w = max(1, int(draw_w))
        safe_h = max(1, int(draw_h))
        display = self.pil_img.resize((safe_w, safe_h), Image.BILINEAR)
        self.tk_img = ImageTk.PhotoImage(display)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_img)

        # Re-draw ROI rectangles if exist
        for idx, roi in enumerate(self.rois):
            bbox = roi.get_bbox()
            if bbox:
                x0, y0, x1, y1 = bbox
                width = 3 if idx == self.active_slot else 2
                color = self.roi_colors[idx % len(self.roi_colors)]
                roi.rect_id = self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=width)

    def _clamp_pan(self, draw_w, draw_h):
        """Clamp pan to valid range when image exceeds canvas.
        When draw > canvas, pan is clamped to show all of the image.
        When draw <= canvas, pan is reset to 0 (centering will be applied in _redraw).
        """
        if draw_w > self.canvas_w:
            # Image wider than canvas: pan_x in [canvas_w - draw_w, 0]
            min_x = self.canvas_w - draw_w
            self.pan_x = max(min_x, min(0, self.pan_x))
        else:
            # Image fits canvas: no pan
            self.pan_x = 0
        
        if draw_h > self.canvas_h:
            # Image taller than canvas: pan_y in [canvas_h - draw_h, 0]
            min_y = self.canvas_h - draw_h
            self.pan_y = max(min_y, min(0, self.pan_y))
        else:
            # Image fits canvas: no pan
            self.pan_y = 0

    def canvas_to_image_coords(self, x, y):
        # Map canvas coords to image coords
        ix = int(max(0, min((x - self.offset_x) / max(self.scale, 1e-6), (self.pil_img.size[0] - 1) if self.pil_img else 0)))
        iy = int(max(0, min((y - self.offset_y) / max(self.scale, 1e-6), (self.pil_img.size[1] - 1) if self.pil_img else 0)))
        return ix, iy

    def roi_image_bbox(self, slot: int = None):
        """Return ROI in original image coordinates as (x, y, w, h), or None.
        If slot is None, uses the active slot.
        """
        if self.pil_img is None:
            return None
        s = self.active_slot if slot is None else int(slot)
        bbox = self.rois[s].get_bbox()
        if not bbox:
            return None
        x0, y0, x1, y1 = bbox
        # Clamp bbox to drawn image rectangle
        draw_w = int(self.pil_img.size[0] * self.scale)
        draw_h = int(self.pil_img.size[1] * self.scale)
        vis_x0 = self.offset_x
        vis_y0 = self.offset_y
        vis_x1 = self.offset_x + draw_w
        vis_y1 = self.offset_y + draw_h
        x0 = max(x0, vis_x0)
        y0 = max(y0, vis_y0)
        x1 = min(x1, vis_x1)
        y1 = min(y1, vis_y1)
        if x1 <= x0 or y1 <= y0:
            return None
        # Map to original image coords
        ix0 = int((x0 - self.offset_x) / self.scale)
        iy0 = int((y0 - self.offset_y) / self.scale)
        ix1 = int((x1 - self.offset_x) / self.scale)
        iy1 = int((y1 - self.offset_y) / self.scale)
        w = max(0, ix1 - ix0)
        h = max(0, iy1 - iy0)
        if w == 0 or h == 0:
            return None
        return (ix0, iy0, w, h)

    def set_roi_from_image_coords(self, slot: int, img_x: int, img_y: int, img_w: int, img_h: int):
        """Set ROI for a slot from original image coordinates (x, y, w, h).
        Converts image coordinates to canvas coordinates and updates the ROI.
        """
        if self.pil_img is None:
            return False
        if slot < 0 or slot > 2:
            return False
        # Ensure we have valid scale/offset (trigger redraw if needed)
        if self.scale == 0:
            self._redraw()
        if self.scale == 0:
            return False
        # Convert image coords to canvas coords
        canvas_x0 = int(self.offset_x + img_x * self.scale)
        canvas_y0 = int(self.offset_y + img_y * self.scale)
        canvas_x1 = int(self.offset_x + (img_x + img_w) * self.scale)
        canvas_y1 = int(self.offset_y + (img_y + img_h) * self.scale)
        # Set ROI
        roi = self.rois[slot]
        roi.set_start(canvas_x0, canvas_y0)
        roi.set_end(canvas_x1, canvas_y1)
        # Remove old rectangle if any
        if roi.rect_id:
            self.canvas.delete(roi.rect_id)
            roi.rect_id = None
        # Redraw to show the new ROI
        self._redraw()
        return True

    def compute_roi_lab(self, slot: int = None):
        if self.np_img is None:
            return (None, None, None)
        roi = self.roi_image_bbox(slot)
        if not roi:
            return (None, None, None)
        x, y, w, h = roi
        rgb_roi = self.np_img[y:y+h, x:x+w, :]
        return rgb_to_lab_mean(rgb_roi)

    def compute_all_roi_lab(self):
        vals = []
        for i in range(3):
            vals.append(self.compute_roi_lab(i))
        return vals

    def set_lab_label(self, slot: int, L, a, b):
        def fmt(v):
            return "-" if v is None else f"{v:.2f}"
        self.lab_lbls[slot].config(text=f"ROI{slot+1}  L*: {fmt(L)}, a*: {fmt(a)}, b*: {fmt(b)}")

    def set_all_lab_labels(self, labs):
        for i in range(3):
            L, a, b = labs[i] if i < len(labs) and labs[i] is not None else (None, None, None)
            self.set_lab_label(i, L, a, b)

    def set_active_slot(self, slot: int):
        self.active_slot = max(0, min(2, int(slot)))
        self._redraw()

    # Event handlers
    def _on_press(self, event):
        if self.pil_img is None:
            return
        roi = self.rois[self.active_slot]
        roi.set_start(event.x, event.y)
        # Remove old rectangle if any
        if roi.rect_id:
            self.canvas.delete(roi.rect_id)
            roi.rect_id = None

    def _on_drag(self, event):
        if self.pil_img is None:
            return
        roi = self.rois[self.active_slot]
        if roi.start is None:
            return
        roi.set_end(event.x, event.y)
        bbox = roi.get_bbox()
        if bbox:
            if roi.rect_id is None:
                color = self.roi_colors[self.active_slot % len(self.roi_colors)]
                roi.rect_id = self.canvas.create_rectangle(*bbox, outline=color, width=3)
            else:
                self.canvas.coords(roi.rect_id, *bbox)

    def _on_release(self, event):
        if self.pil_img is None:
            return
        roi = self.rois[self.active_slot]
        if roi.start is None:
            return
        roi.set_end(event.x, event.y)
        # Final draw update is already handled in _on_drag
        if callable(self.on_roi_changed):
            self.on_roi_changed()

    # Zoom interactions
    def _on_zoom_slider(self):
        if self.pil_img is None:
            return
        new_zoom = max(0.25, min(4.0, float(self._zoom_var.get()) / 100.0))
        self.zoom = new_zoom
        self._zoom_lbl.config(text=f"{int(self.zoom*100)}%")
        self._redraw()

    def _zoom_button(self, zoom_in: bool):
        # Zoom around canvas center
        delta = 120 if zoom_in else -120
        self._on_mousewheel(type('obj', (), {'delta': delta, 'x': self.canvas_w//2, 'y': self.canvas_h//2}))

    def _on_mousewheel_linux(self, direction):
        # direction: +1 up (zoom in), -1 down (zoom out)
        delta = 120 * direction
        self._on_mousewheel(type('obj', (), {'delta': delta, 'x': self.canvas_w//2, 'y': self.canvas_h//2}))

    def _on_mousewheel(self, event):
        """Zoom around cursor position, preserving focus-point stability."""
        if self.pil_img is None:
            return
        # Zoom around cursor position
        old_zoom = self.zoom
        zoom_step = 1.1 if event.delta > 0 else 1/1.1
        new_zoom = max(0.25, min(4.0, old_zoom * zoom_step))
        if abs(new_zoom - old_zoom) < 1e-6:
            return
        # Keep cursor focus point stable: remember which image pixel is under cursor
        cx, cy = event.x, event.y
        img_x_before = (cx - self.offset_x) / max(self.scale, 1e-6)
        img_y_before = (cy - self.offset_y) / max(self.scale, 1e-6)
        # Apply new zoom
        self.zoom = new_zoom
        self._zoom_var.set(self.zoom * 100.0)
        img_w, img_h = self.pil_img.size
        draw_w, draw_h = self._compute_scale(img_w, img_h)
        self.scale = self.zoom
        # Compute new offset so that the same image point stays under cursor
        new_offset_x = int(cx - img_x_before * self.scale)
        new_offset_y = int(cy - img_y_before * self.scale)
        # Set pan based on whether image exceeds canvas:
        # If draw > canvas, pan = new_offset (will be clamped)
        # If draw <= canvas, pan = 0 (centered offset will be computed in _redraw)
        if draw_w > self.canvas_w:
            self.pan_x = new_offset_x
        else:
            self.pan_x = 0
        
        if draw_h > self.canvas_h:
            self.pan_y = new_offset_y
        else:
            self.pan_y = 0
        # Clamp pan to valid ranges
        self._clamp_pan(draw_w, draw_h)
        # Redraw with clamped pan values
        self._redraw()

    # Pan interactions
    def _on_pan_start(self, event):
        if self.pil_img is None:
            return
        self._panning = True
        self._pan_start = (event.x, event.y)

    def _on_pan_move(self, event):
        if not self._panning or self.pil_img is None or self._pan_start is None:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self._pan_start = (event.x, event.y)
        self.pan_x += dx
        self.pan_y += dy
        # Clamp and redraw
        img_w, img_h = self.pil_img.size
        draw_w, draw_h = self._compute_scale(img_w, img_h)
        self._clamp_pan(draw_w, draw_h)
        self._redraw()

    def _on_pan_end(self, event):
        self._panning = False
        self._pan_start = None

    def reset_view(self):
        """Reset zoom to 1.0 (100% = 1:1 pixel mapping) and clear pan."""
        if self.pil_img is None:
            return
        self.zoom = 1.0
        self._zoom_var.set(100.0)
        self._zoom_lbl.config(text="100%")
        self.pan_x = 0
        self.pan_y = 0
        self._redraw()

    def fit_to_screen(self):
        """Compute and apply zoom to fit entire image to canvas.
        zoom = min(canvas_w / img_w, canvas_h / img_h)
        """
        if self.pil_img is None:
            return
        img_w, img_h = self.pil_img.size
        if img_w == 0 or img_h == 0:
            return
        # Compute zoom to fit image to canvas
        self.zoom = min(self.canvas_w / img_w, self.canvas_h / img_h)
        self._zoom_var.set(self.zoom * 100.0)
        self._zoom_lbl.config(text=f"{int(self.zoom*100)}%")
        self.pan_x = 0
        self.pan_y = 0
        self._redraw()


class App:
    def __init__(self, root):
        self.root = root
        root.title("Image Lab Comparator")

        self.root.minsize(1100, 700)


        # Root grid: column 0 images, column 1 controls
        self.main = ttk.Frame(root)
        self.main.pack(fill="both", expand=True)
        self.main.columnconfigure(0, weight=1)
        self.main.columnconfigure(1, weight=0)
        self.main.rowconfigure(0, weight=1)

        images_frame = ttk.Frame(self.main, padding=8)
        images_frame.grid(row=0, column=0, sticky="nsew")

        self.left_panel = ImagePanel(images_frame, title="Reference Image")
        self.left_panel.pack(side="left", padx=(0, 8))

        self.right_panel = ImagePanel(images_frame, title="Test Image (Adjustable)")
        self.right_panel.pack(side="left")

        # Fixed-width sidebar with scrollable canvas
        sidebar = ttk.Frame(self.main, padding=(0, 8, 8, 8), width=360)
        sidebar.grid(row=0, column=1, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(0, weight=1)

        side_canvas = tk.Canvas(sidebar, borderwidth=0, highlightthickness=0)
        scroll_y = ttk.Scrollbar(sidebar, orient="vertical", command=side_canvas.yview)
        side_canvas.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")

        self.controls = ttk.Frame(side_canvas)
        self._controls_win = side_canvas.create_window((0, 0), window=self.controls, anchor="nw")
        side_canvas.configure(yscrollcommand=scroll_y.set)

        def _on_controls_configure(event):
            side_canvas.configure(scrollregion=side_canvas.bbox("all"))

        def _on_canvas_configure(event):
            try:
                side_canvas.itemconfigure(self._controls_win, width=event.width)
            except Exception:
                pass

        self.controls.bind("<Configure>", _on_controls_configure)
        side_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            side_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        side_canvas.bind("<Enter>", lambda e: side_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        side_canvas.bind("<Leave>", lambda e: side_canvas.unbind_all("<MouseWheel>"))

        self.sample_mgr = SampleManager("./dataset")
        self.save_root_var = tk.StringVar(value=self.sample_mgr.root_dir)

        # Load buttons
        load_frame = ttk.LabelFrame(self.controls, text="Load Images")
        load_frame.pack(fill="x", pady=(0, 8))
        ttk.Button(load_frame, text="Load Reference", command=self.load_reference).pack(fill="x", padx=6, pady=(6, 3))
        ttk.Button(load_frame, text="Load Test", command=self.load_test).pack(fill="x", padx=6, pady=(0, 6))

        dataset_frame = ttk.LabelFrame(self.controls, text="Dataset Root")
        dataset_frame.pack(fill="x", padx=6, pady=(6, 0))
        root_row = ttk.Frame(dataset_frame)
        root_row.pack(fill="x", padx=6, pady=4)
        ttk.Label(root_row, text="Root:").pack(side="left")
        root_entry = ttk.Entry(root_row, textvariable=self.save_root_var)
        root_entry.pack(side="left", fill="x", expand=True, padx=6)

        def _apply_root():
            self._ensure_dataset_root()

        def _browse_root():
            path = filedialog.askdirectory(initialdir=self.save_root_var.get() or ".")
            if path:
                self.save_root_var.set(path)
                _apply_root()

        ttk.Button(root_row, text="Browse…", command=_browse_root).pack(side="left")
        ttk.Button(root_row, text="Apply", command=_apply_root).pack(side="left", padx=(4, 0))

        # Lux Meter inputs
        lux_frame = ttk.LabelFrame(self.controls, text="Lux Meter")
        lux_frame.pack(fill="x", padx=6, pady=(6, 0))

        self.ref_lux_min = tk.StringVar()
        self.ref_lux_max = tk.StringVar()
        self.test_lux_min = tk.StringVar()
        self.test_lux_max = tk.StringVar()

        row = 0
        ttk.Label(lux_frame, text="Ref min").grid(row=row, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(lux_frame, textvariable=self.ref_lux_min, width=8).grid(row=row, column=1, padx=4, pady=2)
        ttk.Label(lux_frame, text="Ref max").grid(row=row, column=2, sticky="w", padx=4, pady=2)
        ttk.Entry(lux_frame, textvariable=self.ref_lux_max, width=8).grid(row=row, column=3, padx=4, pady=2)

        row += 1
        ttk.Label(lux_frame, text="Test min").grid(row=row, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(lux_frame, textvariable=self.test_lux_min, width=8).grid(row=row, column=1, padx=4, pady=2)
        ttk.Label(lux_frame, text="Test max").grid(row=row, column=2, sticky="w", padx=4, pady=2)
        ttk.Entry(lux_frame, textvariable=self.test_lux_max, width=8).grid(row=row, column=3, padx=4, pady=2)

        # Lighting tags (multi-select per request)
        lighting_tags_frame = ttk.LabelFrame(self.controls, text="Lighting Tags (multi-select)")
        lighting_tags_frame.pack(fill="x", pady=(0, 8))
        self.lighting_checks = {}
        tag_names = [
            "daylight",
            "general light cold",
            "general light warm",
            "dental chair light",
            "special photolight",
            "other",
        ]
        for name in tag_names:
            var = tk.BooleanVar(value=False)
            chk = ttk.Checkbutton(lighting_tags_frame, text=name, variable=var)
            chk.pack(anchor="w", padx=6, pady=1)
            self.lighting_checks[name] = var

        # ROI Tools
        roi_tools_frame = ttk.LabelFrame(self.controls, text="ROI Tools")
        roi_tools_frame.pack(fill="x", pady=(0, 8))
        self.copy_rois_btn = ttk.Button(roi_tools_frame, text="Copy ROIs →", command=self.copy_rois_from_reference)
        self.copy_rois_btn.pack(fill="x", padx=6, pady=6)
        # Initially disable if no ROIs defined
        self.copy_rois_btn.config(state="disabled")

        # Slider controls
        sliders = ttk.LabelFrame(self.controls, text="Image Adjustments")
        sliders.pack(fill="x", pady=(0, 8))

        # Single white balance (temperature) and advanced per-channel gains
        self.wb_temp = tk.IntVar(value=0)       # -100..100 (cool .. warm)
        self.r_gain = tk.DoubleVar(value=1.0)   # 0.5..1.5 (advanced)
        self.b_gain = tk.DoubleVar(value=1.0)   # 0.5..1.5 (advanced)
        self.brightness = tk.IntVar(value=0)   # -100..100
        self.contrast = tk.IntVar(value=0)     # -50..50
        self.exposure_ev = tk.DoubleVar(value=0.0)  # -2..+2 EV

        def slider_row(parent, text, var, from_, to_, step, cmd):
            row = ttk.Frame(parent)
            row.pack(fill="x", padx=6, pady=3)
            ttk.Label(row, text=text, width=12).pack(side="left")
            scale = ttk.Scale(row, from_=from_, to=to_, orient="horizontal", variable=var, command=lambda v: cmd())
            scale.pack(side="left", fill="x", expand=True, padx=6)
            val_lbl = ttk.Label(row)

            def clamp(val):
                return max(from_, min(to_, val))

            def update_label():
                try:
                    val = var.get()
                    if isinstance(var, tk.IntVar):
                        val_lbl.config(text=f"{int(val)}")
                    else:
                        val_lbl.config(text=f"{val:.2f}")
                except Exception:
                    pass
            # Track variable changes to update label
            var.trace_add("write", lambda *args: update_label())
            val_lbl.pack(side="left", padx=(6, 0))

            # Nudge buttons
            def nudge(d):
                try:
                    cur = var.get()
                    new = cur + d
                    if isinstance(var, tk.IntVar):
                        new = int(round(new))
                    var.set(clamp(new))
                    cmd()
                except Exception:
                    pass
            ttk.Button(row, text="-", width=3, command=lambda: nudge(-step)).pack(side="left", padx=(6, 0))
            ttk.Button(row, text="+", width=3, command=lambda: nudge(step)).pack(side="left", padx=(3, 0))

        slider_row(sliders, "White balance", self.wb_temp, -100, 100, 1, self.on_adjustments_changed)
        slider_row(sliders, "Red gain", self.r_gain, 0.5, 1.5, 0.01, self.on_adjustments_changed)
        slider_row(sliders, "Blue gain", self.b_gain, 0.5, 1.5, 0.01, self.on_adjustments_changed)
        slider_row(sliders, "Brightness", self.brightness, -100, 100, 1, self.on_adjustments_changed)
        slider_row(sliders, "Contrast", self.contrast, -50, 50, 1, self.on_adjustments_changed)
        slider_row(sliders, "Exposure EV", self.exposure_ev, -2.0, 2.0, 0.1, self.on_adjustments_changed)

        # Delta E Metrics
        deltae_frame = ttk.LabelFrame(self.controls, text="Delta E Metrics")
        deltae_frame.pack(fill="x", pady=(0, 8))

        # Delta E display
        self.deltae_label = ttk.Label(deltae_frame, text="ΔE76: -", font=("Segoe UI", 10, "bold"))
        self.deltae_label.pack(fill="x", padx=6, pady=(6, 8))
        self.delta_components_label = ttk.Label(deltae_frame, text="ΔL: -, Δa: -, Δb: -")
        self.delta_components_label.pack(fill="x", padx=6, pady=(0, 8))

        # ΔE display mode (DE76 or DE00)
        self.delta_mode = tk.StringVar(value="DE76")
        mode_frame = ttk.LabelFrame(deltae_frame, text="ΔE Mode (display)")
        mode_frame.pack(fill="x", padx=6, pady=(6, 6))
        ttk.Radiobutton(mode_frame, text="ΔE76 (CIE76)", variable=self.delta_mode, value="DE76", command=self.compute_both_lab).pack(anchor="w", padx=6, pady=2)
        ttk.Radiobutton(mode_frame, text="ΔE00 (CIEDE2000)", variable=self.delta_mode, value="DE00", command=self.compute_both_lab).pack(anchor="w", padx=6, pady=2)

        # Per-ROI ΔE labels and overall metrics
        self.deltae_roi_lbls = [
            ttk.Label(deltae_frame, text="ROI1 ΔE76: -"),
            ttk.Label(deltae_frame, text="ROI2 ΔE76: -"),
            ttk.Label(deltae_frame, text="ROI3 ΔE76: -"),
        ]
        for lbl in self.deltae_roi_lbls:
            lbl.pack(fill="x", padx=6, pady=(0, 2))
        self.deltae_mean_label_76 = ttk.Label(deltae_frame, text="Mean ΔE76 (3 ROIs): -", font=("Segoe UI", 9, "bold"))
        self.deltae_pooled_label_76 = ttk.Label(deltae_frame, text="Pooled ΔE76: -", font=("Segoe UI", 9, "bold"))
        self.deltae_mean_label_00 = ttk.Label(deltae_frame, text="Mean ΔE00 (3 ROIs): -", font=("Segoe UI", 9, "bold"))
        self.deltae_pooled_label_00 = ttk.Label(deltae_frame, text="Pooled ΔE00: -", font=("Segoe UI", 9, "bold"))
        self.deltae_mean_label_76.pack(fill="x", padx=6, pady=(6, 0))
        self.deltae_pooled_label_76.pack(fill="x", padx=6, pady=(0, 6))
        self.deltae_mean_label_00.pack(fill="x", padx=6, pady=(6, 0))
        self.deltae_pooled_label_00.pack(fill="x", padx=6, pady=(0, 8))

        # Colorimeter Reference
        colorimeter_frame = ttk.LabelFrame(self.controls, text="Colorimeter Comparison")
        colorimeter_frame.pack(fill="x", padx=6, pady=(0, 6))
        
        self.colorimeter_L = tk.StringVar()
        self.colorimeter_a = tk.StringVar()
        self.colorimeter_b = tk.StringVar()
        self.colorimeter_selection = tk.StringVar(value="Test ROI 1")  # default selection
        
        # L*, a*, b* inputs
        ttk.Label(colorimeter_frame, text="Colorimeter Reading:").pack(anchor="w", padx=6, pady=(4, 2))
        lab_row = ttk.Frame(colorimeter_frame)
        lab_row.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(lab_row, text="L*").pack(side="left", padx=(0, 4))
        ttk.Entry(lab_row, textvariable=self.colorimeter_L, width=8).pack(side="left", padx=(0, 8))
        ttk.Label(lab_row, text="a*").pack(side="left", padx=(0, 4))
        ttk.Entry(lab_row, textvariable=self.colorimeter_a, width=8).pack(side="left", padx=(0, 8))
        ttk.Label(lab_row, text="b*").pack(side="left", padx=(0, 4))
        ttk.Entry(lab_row, textvariable=self.colorimeter_b, width=8).pack(side="left")
        
        # ROI selection
        roi_select_row = ttk.Frame(colorimeter_frame)
        roi_select_row.pack(fill="x", padx=6, pady=(2, 4))
        ttk.Label(roi_select_row, text="Compare With:").pack(side="left")
        roi_options = ["Reference ROI 1", "Reference ROI 2", "Reference ROI 3", "Test ROI 1", "Test ROI 2", "Test ROI 3"]
        self.roi_combobox = ttk.Combobox(roi_select_row, textvariable=self.colorimeter_selection, values=roi_options, state="readonly", width=15)
        self.roi_combobox.pack(side="left", padx=(6, 0))
        
        # Compare button
        ttk.Button(colorimeter_frame, text="Compare to Colorimeter", command=self.compare_to_colorimeter).pack(fill="x", padx=6, pady=(0, 6))
        
        # ΔE results labels
        self.colorimeter_de76_label = ttk.Label(colorimeter_frame, text="ΔE76 (Selected ROI vs Colorimeter): -")
        self.colorimeter_de76_label.pack(fill="x", padx=6, pady=(0, 2))
        self.colorimeter_de00_label = ttk.Label(colorimeter_frame, text="ΔE00 (Selected ROI vs Colorimeter): -")
        self.colorimeter_de00_label.pack(fill="x", padx=6, pady=(0, 6))

        actions = ttk.LabelFrame(self.controls, text="Action Buttons")
        actions.pack(fill="x", pady=(0, 8))
        ttk.Button(actions, text="Reset Adjustments", command=self.reset_adjustments).pack(fill="x", padx=6, pady=(6, 3))
        ttk.Button(actions, text="Compute LAB (Both)", command=self.compute_both_lab).pack(fill="x", padx=6, pady=(6, 3))
        ttk.Button(actions, text="Match L*", command=self.match_lightness).pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(actions, text="Save Data", command=self.save_data).pack(fill="x", padx=6, pady=(0, 6))

        # Status
        self.status = tk.StringVar(value="Tip: click a panel to focus it, then press 1/2/3 to switch ROI slots. Or use ROI1/2/3 buttons under each image.")
        ttk.Label(self.controls, textvariable=self.status, wraplength=260).pack(fill="x", pady=(4, 0))

        # Image paths
        self.ref_path = None
        self.test_path = None

        # Store original test image as numpy for reapplying adjustments
        self.test_np_original = None

        # Wire ROI change callbacks for live updates
        def left_roi_changed():
            self.on_adjustments_changed()
            self._update_copy_rois_button_state()
        self.left_panel.on_roi_changed = left_roi_changed
        self.right_panel.on_roi_changed = self.on_adjustments_changed

    def _set_deltae_text(self, dE):
        if dE is None:
            self.deltae_label.configure(text="ΔE76: -", foreground="black")
            return
        txt = f"ΔE76: {dE:.2f}"
        color = "green" if dE < 2 else ("orange" if dE < 5 else "red")
        self.deltae_label.configure(text=txt, foreground=color)

    @staticmethod
    def _parse_float_or_none(s):
        """Parse a string to float, handling various formats including scientific notation and multiplier formats.
        
        Supports:
        - Regular numbers: "55.3", "123"
        - Comma as decimal separator: "55,3"
        - Scientific notation: "5.4e2", "5.59e+01", "5.59E+01", "5.59e-02"
        - Multiplier formats: "55.3x10", "55.3x100" (with flexible spacing)
        - All multiplier symbol variants: x, X, ×, х, Х (CYRILLIC)
        - Empty strings or invalid values return None
        """
        if not s or not isinstance(s, str):
            return None
        
        s = s.strip()
        if not s:
            return None
        
        try:
            # Normalize decimal separator: replace comma with dot
            s = s.replace(',', '.')
            
            # Normalize all multiplier variants to lowercase 'x':
            # - Latin: 'x', 'X'
            # - Unicode: '×' (multiplication sign, U+00D7)
            # - Cyrillic: 'х', 'Х' (CYRILLIC SMALL/CAPITAL LETTER HA, U+0445/U+0425)
            s = s.replace("×", "x").replace("X", "x").replace("х", "x").replace("Х", "x")
            
            # Check for multiplier patterns: "x10" or "x100" with flexible spacing
            # Regex: 'x' followed by optional spaces, then (10|100), then optional trailing spaces
            import re
            match = re.search(r'x\s*(10|100)\s*$', s, re.IGNORECASE)
            if match:
                # Extract multiplier value (10 or 100) from the regex capture group
                multiplier = int(match.group(1))
                # Extract base number before the multiplier pattern
                base_str = s[:match.start()].strip()
                if base_str:
                    base_val = float(base_str)
                    return base_val * multiplier
            
            # Try direct float conversion (handles regular numbers and scientific notation like "5.4e2", "5.59e+01")
            return float(s)
        except (ValueError, TypeError):
            return None

    def _ensure_dataset_root(self):
        path = self.save_root_var.get().strip() or "./dataset"
        self.sample_mgr.set_root(path)
        self.save_root_var.set(self.sample_mgr.root_dir)
        return self.sample_mgr.root_dir

    @staticmethod
    def _roi_dict(roi):
        if roi is None:
            return None
        return {"x": int(roi[0]), "y": int(roi[1]), "w": int(roi[2]), "h": int(roi[3])}

    @staticmethod
    def _lab_dict(lab):
        if not lab or lab[0] is None:
            return None
        L, a, b = lab
        return {"L": float(L), "a": float(a), "b": float(b)}

    def get_current_adjusted_test_rgb(self):
        arr = self._adjusted_test_np()
        if arr is None:
            return None
        return arr.copy()

    def _format_de(self, de):
        return "-" if de is None else f"{de:.2f}"

    def _color_for_de(self, de):
        if de is None:
            return "black"
        return "green" if de < 2 else ("orange" if de < 5 else "red")

    def _set_per_roi_and_overall(self, deltaEs, comps, mean_de, pooled_de):
        # Legacy helper no longer used for overall; keep for compatibility if called elsewhere.
        mode = self.delta_mode.get()
        suffix = "00" if mode == "DE00" else "76"
        for i, de in enumerate(deltaEs):
            txt = f"ROI{i+1} ΔE{suffix}: {self._format_de(de)}"
            self.deltae_roi_lbls[i].configure(text=txt, foreground=self._color_for_de(de))
        # Set only 76 labels for pooled/mean for backward-compat
        self.deltae_mean_label_76.configure(
            text=f"Mean ΔE76 (3 ROIs): {self._format_de(mean_de)}",
            foreground=self._color_for_de(mean_de),
        )
        self.deltae_pooled_label_76.configure(
            text=f"Pooled ΔE76: {self._format_de(pooled_de)}",
            foreground=self._color_for_de(pooled_de),
        )

    def _set_overall_pairs(self, mean76, pooled76, mean00, pooled00):
        self.deltae_mean_label_76.configure(
            text=f"Mean ΔE76 (3 ROIs): {self._format_de(mean76)}",
            foreground=self._color_for_de(mean76),
        )
        self.deltae_pooled_label_76.configure(
            text=f"Pooled ΔE76: {self._format_de(pooled76)}",
            foreground=self._color_for_de(pooled76),
        )
        self.deltae_mean_label_00.configure(
            text=f"Mean ΔE00 (3 ROIs): {self._format_de(mean00)}",
            foreground=self._color_for_de(mean00),
        )
        # pooled00 approximated via pooled components as ΔE76
        self.deltae_pooled_label_00.configure(
            text=f"Pooled ΔE00: {self._format_de(pooled00)}",
            foreground=self._color_for_de(pooled00),
        )

    def _update_per_roi_labels(self, des, mode):
        suffix = "00" if mode == "DE00" else "76"
        for i, de in enumerate(des):
            txt = f"ROI{i+1} ΔE{suffix}: {self._format_de(de)}"
            self.deltae_roi_lbls[i].configure(text=txt, foreground=self._color_for_de(de))

    def _set_delta_components(self, comps):
        if not comps:
            self.delta_components_label.configure(text="ΔL: -, Δa: -, Δb: -")
            return
        dL, da, db = comps
        self.delta_components_label.configure(text=f"ΔL: {dL:.2f}, Δa: {da:.2f}, Δb: {db:.2f}")

    # -- Image loading --
    def load_reference(self):
        path = filedialog.askopenfilename(title="Select Reference Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.heic;*.heif")])
        if not path:
            return
        try:
            pil = Image.open(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
            return
        self.ref_path = path
        self.left_panel.load_image(pil)
        self.left_panel.set_all_lab_labels([(None, None, None)] * 3)
        self._update_copy_rois_button_state()
        self.status.set(f"Loaded reference: {os.path.basename(path)}")

    def load_test(self):
        path = filedialog.askopenfilename(title="Select Test Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.heic;*.heif")])
        if not path:
            return
        try:
            pil = Image.open(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
            return
        self.test_path = path
        np_img = np_from_pil(pil)
        self.test_np_original = np_img
        # Apply current adjustments (initially identity) and draw
        adjusted = self._adjusted_test_np()
        self.right_panel.update_from_np(adjusted)
        self.right_panel.set_all_lab_labels([(None, None, None)] * 3)
        self.status.set(f"Loaded test: {os.path.basename(path)}")

    def _adjusted_test_np(self):
        if self.test_np_original is None:
            return None
        # Map single WB slider (-100..100) to symmetric red/blue gains (0.5..1.5)
        t = float(self.wb_temp.get()) / 100.0
        r_temp = 1.0 + 0.5 * t
        b_temp = 1.0 - 0.5 * t
        r_eff = float(self.r_gain.get()) * r_temp
        b_eff = float(self.b_gain.get()) * b_temp
        arr = apply_adjustments(
            self.test_np_original,
            r_eff,
            b_eff,
            int(self.brightness.get()),
            int(self.contrast.get()),
        )
        # Exposure in linear domain
        ev = float(self.exposure_ev.get())
        arr = apply_exposure_rgb(arr, ev)
        return arr

    # -- Adjustments --
    def on_adjustments_changed(self):
        try:
            if self.test_np_original is None:
                return
            adjusted = self._adjusted_test_np()
            self.right_panel.update_from_np(adjusted)
            # Recompute LAB across all 3 ROIs and update labels
            labs_ref = self.left_panel.compute_all_roi_lab()
            labs_test = self.right_panel.compute_all_roi_lab()
            self.left_panel.set_all_lab_labels(labs_ref)
            self.right_panel.set_all_lab_labels(labs_test)

            # Active ROI detail (by selected mode)
            s = self.left_panel.active_slot
            lab_ref = labs_ref[s] if labs_ref[s][0] is not None else None
            lab_tst = labs_test[s] if labs_test[s][0] is not None else None
            if self.delta_mode.get() == "DE00":
                dE_active = delta_e_ciede2000(lab_ref, lab_tst)
            else:
                dE_active = delta_e_cie76(lab_ref, lab_tst)
            self._set_deltae_text(dE_active)
            self._set_delta_components(delta_components(lab_ref, lab_tst))

            # Per-ROI + overall for both metrics
            de76_list, mean76, pooled76 = pairwise_deltaEs(labs_ref, labs_test, metric="DE76")
            de00_list, mean00, pooled00 = pairwise_deltaEs(labs_ref, labs_test, metric="DE00")
            current_list = de00_list if self.delta_mode.get() == "DE00" else de76_list
            self._update_per_roi_labels(current_list, self.delta_mode.get())
            self._set_overall_pairs(mean76, pooled76, mean00, pooled00)
        except Exception as exc:
            self.status.set(f"Update error: {exc}")

    def reset_adjustments(self):
        # Restore defaults
        self.wb_temp.set(0)
        self.r_gain.set(1.0)
        self.b_gain.set(1.0)
        self.brightness.set(0)
        self.contrast.set(0)
        self.on_adjustments_changed()

    def _update_copy_rois_button_state(self):
        """Update the state of the Copy ROIs button based on whether ROIs exist on reference panel."""
        if self.left_panel.pil_img is None:
            self.copy_rois_btn.config(state="disabled")
            return
        # Check if any ROI is defined on reference panel
        has_roi = False
        for i in range(3):
            roi_bbox = self.left_panel.roi_image_bbox(i)
            if roi_bbox is not None:
                has_roi = True
                break
        self.copy_rois_btn.config(state="normal" if has_roi else "disabled")

    def copy_rois_from_reference(self):
        """Copy ROIs from reference image panel to test image panel."""
        if self.right_panel.pil_img is None:
            self.status.set("Please load test image first.")
            return
        if self.left_panel.pil_img is None:
            self.status.set("Please load reference image first.")
            return
        
        copied_count = 0
        for slot in range(3):
            # Get ROI from reference panel in image coordinates
            ref_roi = self.left_panel.roi_image_bbox(slot)
            if ref_roi is None:
                continue  # Skip if no ROI defined for this slot
            
            img_x, img_y, img_w, img_h = ref_roi
            # Set ROI on test panel
            if self.right_panel.set_roi_from_image_coords(slot, img_x, img_y, img_w, img_h):
                copied_count += 1
        
        if copied_count == 0:
            self.status.set("No ROIs found on reference image to copy.")
            return
        
        # Update LAB values and trigger adjustments update
        self.on_adjustments_changed()
        self.status.set(f"Copied {copied_count} ROI(s) from reference to test image.")

    # -- LAB computation --
    def compute_both_lab(self):
        if cv2 is None:
            messagebox.showwarning("OpenCV not found", "OpenCV (cv2) is required for LAB computation.")
            return
        labs_ref = self.left_panel.compute_all_roi_lab()
        labs_test = self.right_panel.compute_all_roi_lab()
        self.left_panel.set_all_lab_labels(labs_ref)
        self.right_panel.set_all_lab_labels(labs_test)

        # Active ROI detail, by mode
        s = self.left_panel.active_slot
        lab_ref = labs_ref[s] if labs_ref[s][0] is not None else None
        lab_tst = labs_test[s] if labs_test[s][0] is not None else None
        if self.delta_mode.get() == "DE00":
            dE_active = delta_e_ciede2000(lab_ref, lab_tst)
        else:
            dE_active = delta_e_cie76(lab_ref, lab_tst)
        self._set_deltae_text(dE_active)
        self._set_delta_components(delta_components(lab_ref, lab_tst))

        # Per-ROI + overall for both metrics
        de76_list, mean76, pooled76 = pairwise_deltaEs(labs_ref, labs_test, metric="DE76")
        de00_list, mean00, pooled00 = pairwise_deltaEs(labs_ref, labs_test, metric="DE00")
        current_list = de00_list if self.delta_mode.get() == "DE00" else de76_list
        self._update_per_roi_labels(current_list, self.delta_mode.get())
        self._set_overall_pairs(mean76, pooled76, mean00, pooled00)
        self.status.set("Computed LAB and ΔE for all 3 ROIs.")

    def compare_to_colorimeter(self):
        # Get colorimeter LAB values
        L_ref = self._parse_float_or_none(self.colorimeter_L.get())
        a_ref = self._parse_float_or_none(self.colorimeter_a.get())
        b_ref = self._parse_float_or_none(self.colorimeter_b.get())
        
        if L_ref is None or a_ref is None or b_ref is None:
            self.status.set("Please enter valid L*, a*, and b* values for colorimeter reference.")
            return
        
        lab_colorimeter = (L_ref, a_ref, b_ref)
        
        # Parse selection
        selection = self.colorimeter_selection.get()
        if "Reference" in selection:
            panel = self.left_panel
            source_name = "Reference"
        else:
            panel = self.right_panel
            source_name = "Test"
        
        roi_index = int(selection.split()[-1]) - 1  # Extract ROI number (1-3) and convert to 0-2
        
        # Get selected ROI LAB
        lab_roi = panel.compute_roi_lab(roi_index)
        if lab_roi[0] is None:
            self.status.set(f"Please select {source_name} ROI {roi_index + 1} on the image first.")
            return
        
        # Compute ΔE76 and ΔE00
        de76 = delta_e_cie76(lab_roi, lab_colorimeter)
        de00 = delta_e_ciede2000(lab_roi, lab_colorimeter)
        
        # Update labels with color coding
        self._set_colorimeter_de_label(self.colorimeter_de76_label, de76, "ΔE76 (Selected ROI vs Colorimeter)")
        self._set_colorimeter_de_label(self.colorimeter_de00_label, de00, "ΔE00 (Selected ROI vs Colorimeter)")
        
        self.status.set(f"Compared {source_name} ROI{roi_index + 1} to colorimeter: ΔE76={de76:.2f}, ΔE00={de00:.2f}")

    def _set_colorimeter_de_label(self, label, de_value, prefix):
        if de_value is None:
            label.configure(text=f"{prefix}: -", foreground="black")
            return
        txt = f"{prefix}: {de_value:.2f}"
        if de_value < 1.0:
            color = "green"
        elif de_value < 2.5:
            color = "orange"
        else:
            color = "red"
        label.configure(text=txt, foreground=color)

    def match_lightness(self):
        # Needs ROIs and both LABs
        Lr, ar, br = self.left_panel.compute_roi_lab()
        Lt, at, bt = self.right_panel.compute_roi_lab()
        if Lr is None or Lt is None:
            self.status.set("Select ROIs on both images first.")
            return
        target = Lr
        ev0 = float(self.exposure_ev.get())
        best_ev = ev0
        best_err = abs(Lt - target)
        # Coarse grid search across -2..2 EV
        for ev in np.linspace(-2.0, 2.0, 33):
            self.exposure_ev.set(ev)
            arr = self._adjusted_test_np()
            self.right_panel.update_from_np(arr)
            Lc, ac, bc = self.right_panel.compute_roi_lab()
            if Lc is None:
                continue
            err = abs(Lc - target)
            if err < best_err:
                best_err, best_ev = err, ev
            if best_err <= 0.1:
                break
        # Apply best EV
        self.exposure_ev.set(best_ev)
        arr = self._adjusted_test_np()
        self.right_panel.update_from_np(arr)
        self.on_adjustments_changed()
        Lc, ac, bc = self.right_panel.compute_roi_lab()
        self.status.set(f"Matched L*: target {target:.2f}, got {Lc:.2f} at EV {best_ev:+.2f}")

    # -- Save --
    def save_data(self):
        if self.left_panel.np_img is None or self.right_panel.np_img is None:
            messagebox.showinfo("Missing images", "Please load both reference and test images first.")
            return

        self._ensure_dataset_root()

        rois_ref = [self.left_panel.roi_image_bbox(i) for i in range(3)]
        rois_test = [self.right_panel.roi_image_bbox(i) for i in range(3)]
        if not any((r is not None and t is not None) for r, t in zip(rois_ref, rois_test)):
            messagebox.showinfo("Missing ROIs", "Please select at least one matching ROI on both images before saving.")
            return

        labs_ref = self.left_panel.compute_all_roi_lab()
        labs_test = self.right_panel.compute_all_roi_lab()

        deltaEs76, comps, mean76_components, pooled76_components = pairwise_deltaE_and_components(labs_ref, labs_test)
        de76_list, mean76, pooled76 = pairwise_deltaEs(labs_ref, labs_test, metric="DE76")
        de00_list, mean00, pooled00 = pairwise_deltaEs(labs_ref, labs_test, metric="DE00")

        selected_tags = [name for name, v in self.lighting_checks.items() if v.get()]

        meta = {
            "lighting_tags": selected_tags,
            "rois_reference": [self._roi_dict(r) for r in rois_ref],
            "rois_test": [self._roi_dict(t) for t in rois_test],
            "labs_reference": [self._lab_dict(l) for l in labs_ref],
            "labs_test": [self._lab_dict(l) for l in labs_test],
            "sliders": {
                "white_balance": int(self.wb_temp.get()),
                "red_gain": float(self.r_gain.get()),
                "blue_gain": float(self.b_gain.get()),
                "brightness": int(self.brightness.get()),
                "contrast": int(self.contrast.get()),
                "exposure_ev": float(self.exposure_ev.get()),
            },
            "lux_ref": {
                "min": self._parse_float_or_none(self.ref_lux_min.get()),
                "max": self._parse_float_or_none(self.ref_lux_max.get()),
            },
            "lux_test": {
                "min": self._parse_float_or_none(self.test_lux_min.get()),
                "max": self._parse_float_or_none(self.test_lux_max.get()),
            },
        }

        meta["deltaE76_per_roi"] = [None if de is None else float(de) for de in deltaEs76]
        meta["overall"] = {
            "mean_deltaE76": None if mean76_components is None else float(mean76_components),
            "pooled_deltaE76": None if pooled76_components is None else float(pooled76_components),
            "delta_components_per_roi": [
                None if c is None else {"dL": float(c[0]), "da": float(c[1]), "db": float(c[2])}
                for c in comps
            ],
        }
        meta["deltaE"] = {
            "per_roi": {
                "DE76": [None if v is None else float(v) for v in de76_list],
                "DE00": [None if v is None else float(v) for v in de00_list],
            },
            "overall": {
                "mean_DE76": None if mean76 is None else float(mean76),
                "pooled_DE76": None if pooled76 is None else float(pooled76),
                "mean_DE00": None if mean00 is None else float(mean00),
                "pooled_DE00_as_DE76_components": None if pooled00 is None else float(pooled00),
            },
        }

        # Reference image is not saved - only used temporarily for LAB comparison
        test_orig_source = self.test_path if self.test_path else self.test_np_original
        if test_orig_source is None:
            test_orig_source = self.right_panel.np_img
        test_adj_rgb = self.get_current_adjusted_test_rgb()
        if test_adj_rgb is None:
            messagebox.showwarning("No adjusted image", "Unable to render adjusted test image for saving.")
            return

        try:
            sample_id, sample_path = self.sample_mgr.make_sample_folder()
            self.sample_mgr.save_sample(
                sample_path,
                ref_img=None,  # Reference image not saved
                test_orig_img=test_orig_source,
                test_adj_img=test_adj_rgb,
                meta_dict=meta,
            )
        except Exception as exc:
            messagebox.showerror("Save failed", f"Could not save sample:\n{exc}")
            return

        self.status.set(f"Saved sample #{sample_id} to {sample_path}")
        messagebox.showinfo("Saved", f"Sample {sample_id} saved to\n{sample_path}")
def main():
    root = tk.Tk()
    # Themed style tweaks for clarity
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    try:
        root.geometry("1280x800")
    except Exception:
        pass
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

