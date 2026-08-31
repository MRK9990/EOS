from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import joblib
import cv2
from PIL import Image, ImageTk
import exifread
import pillow_heif
pillow_heif.register_heif_opener()

# ============================================================
# EOS PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "eos_model_v1.pkl"

CROP_WIDTH = 600
CROP_HEIGHT = 500
CROP_ASPECT_RATIO = CROP_WIDTH / CROP_HEIGHT  # 6/5 = 1.2

# ===== Utility Functions =====

def rgb_to_lab_mean(rgb_roi):
    """Compute mean LAB from RGB ROI."""
    if rgb_roi is None or rgb_roi.size == 0:
        return None
    bgr = rgb_roi[..., ::-1].astype(np.float32) / 255.0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = float(np.mean(lab[..., 0]))
    a = float(np.mean(lab[..., 1]))
    b = float(np.mean(lab[..., 2]))
    return L, a, b


def extract_exif_data(image_path):
    """Extract EXIF metadata from image."""
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)

        iso = tags.get("EXIF ISOSpeedRatings", 0)
        exposure = tags.get("EXIF ExposureTime", 0)
        fnumber = tags.get("EXIF FNumber", 0)
        datetime_original = tags.get("EXIF DateTimeOriginal", None)

        if iso != 0:
            iso = int(str(iso))

        if exposure != 0:
            num, den = str(exposure).split('/')
            exposure = float(num) / float(den)

        if fnumber != 0:
            num, den = str(fnumber).split('/')
            fnumber = float(num) / float(den)

        hour = 0
        minute = 0

        if datetime_original:
            dt = str(datetime_original)
            time_part = dt.split(" ")[1]
            hour, minute, _ = map(int, time_part.split(":"))

        return iso, exposure, fnumber, hour, minute
    except:
        return 0, 0, 0, 0, 0


def apply_adjustments(rgb, r_gain, b_gain, brightness, contrast):
    """Apply white balance and brightness/contrast adjustments."""
    arr = rgb.astype(np.float32)
    arr[..., 0] *= r_gain
    arr[..., 2] *= b_gain
    alpha = 1.0 + (contrast / 100.0)
    beta = brightness
    arr = arr * alpha + beta
    return np.clip(arr, 0, 255).astype(np.uint8)


def apply_exposure_rgb(rgb, ev):
    """Apply exposure compensation in linear light space."""
    arr = rgb.astype(np.float32) / 255.0
    thr = 0.04045
    lin = np.where(arr <= thr, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    lin *= (2.0 ** float(ev))
    srgb = np.where(lin <= 0.0031308, lin * 12.92,
                    1.055 * (np.clip(lin, 0, 1) ** (1 / 2.4)) - 0.055)
    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)


def crop_with_aspect_ratio(original_np, crop_start, crop_end):
    """
    Crop from original image with 6:5 aspect ratio constraint.
    Returns cropped region and scaling factors for coordinate mapping.
    """
    x0, y0 = crop_start
    x1, y1 = crop_end
    
    # Swap coordinates to ensure proper bounds
    left = min(x0, x1)
    right = max(x0, x1)
    top = min(y0, y1)
    bottom = max(y0, y1)
    
    # Force aspect ratio 6:5
    width = right - left
    height = bottom - top
    
    if width / height > CROP_ASPECT_RATIO:
        height = int(width / CROP_ASPECT_RATIO)
    else:
        width = int(height * CROP_ASPECT_RATIO)
    
    # Clamp to original image bounds
    img_h, img_w = original_np.shape[:2]
    left = max(0, min(left, img_w - width))
    top = max(0, min(top, img_h - height))
    right = min(left + width, img_w)
    bottom = min(top + height, img_h)
    
    # Re-adjust in case of clamping
    width = right - left
    height = bottom - top
    
    cropped = original_np[top:bottom, left:right]
    cropped_resized = cv2.resize(cropped, (CROP_WIDTH, CROP_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    
    return cropped_resized, (left, top, width, height)


# ===== GUI =====

class EOSPredictor:

    def __init__(self, root):
        self.root = root
        self.root.title("EOS Predictor v1")
        self.root.configure(bg="#1e1e1e")

        # Load model
        if not MODEL_PATH.exists():
            messagebox.showerror(
                "EOS Error",
                f"Model not found:\n{MODEL_PATH}"
            )
            self.root.destroy()
            return

        data = joblib.load(MODEL_PATH)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_columns = data["feature_columns"]
        # Image & State
        self.image_path = None
        self.original_np = None       # Full resolution image
        self.cropped_np = None        # 600x500 cropped image
        self.adjusted_np = None       # Display image after prediction
        self.roi = None               # ROI in cropped image coordinates
        self.crop_region = None       # Crop bounds in original image (left, top, width, height)
        
        self.mode = "idle"            # "idle", "crop_mode", "roi_mode"
        self.drag_start = None

        self.setup_ui()

    def setup_ui(self):
        """Setup dark-themed UI with three-column layout."""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background="#1e1e1e", foreground="white")
        style.configure("TButton", background="#333333", foreground="white", padding=5)
        style.configure("TLabel", background="#1e1e1e", foreground="white")
        style.configure("TEntry", fieldbackground="#2b2b2b", foreground="white")
        style.configure("TCheckbutton", background="#1e1e1e", foreground="white")

        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # LEFT CANVAS - Crop/ROI selection
        self.canvas = tk.Canvas(main, bg="#111111", width=CROP_WIDTH, height=CROP_HEIGHT, 
                               highlightbackground="#333333", highlightthickness=2)
        self.canvas.grid(row=0, column=0, rowspan=20, padx=10, pady=10)

        # CENTER CONTROLS
        # Mode label
        self.mode_label = ttk.Label(main, text="IDLE", foreground="#888888")
        self.mode_label.grid(row=0, column=1, sticky="ew", pady=5)

        # Load Image button
        ttk.Button(main, text="Load Image", command=self.load_image).grid(row=1, column=1, sticky="ew", pady=5)

        # Reset button
        self.reset_button = ttk.Button(main, text="Reset Crop", command=self.reset_crop)
        self.reset_button.grid(row=2, column=1, sticky="ew", pady=5)

        # Lux inputs
        ttk.Label(main, text="Lux Min").grid(row=3, column=1, sticky="w")
        self.lux_min = tk.Entry(main, width=15, fg="white", bg="#2b2b2b")
        self.lux_min.grid(row=4, column=1, sticky="ew", pady=3)

        ttk.Label(main, text="Lux Max").grid(row=5, column=1, sticky="w")
        self.lux_max = tk.Entry(main, width=15, fg="white", bg="#2b2b2b")
        self.lux_max.grid(row=6, column=1, sticky="ew", pady=3)

        # ========================================================
        # LIGHTING CONDITIONS
        # ========================================================

        ttk.Label(
            main,
            text="Lighting",
            foreground="#888888"
        ).grid(row=7, column=1, sticky="ew", pady=(10, 5))

        self.tags = {}

        tag_list = [
            "daylight",
            "general light cold",
            "general light warm",
            "dental chair light",
            "special photolight",
            "other"
        ]

        for i, tag in enumerate(tag_list, start=8):
            var = tk.BooleanVar(value=False)
            self.tags[tag] = var

            ttk.Checkbutton(
                main,
                text=tag.replace("_", " ").title(),
                variable=var
            ).grid(
                row=i,
                column=1,
                sticky="w",
                pady=2
            )

        # ========================================================
        # ENVIRONMENT CONDITIONS
        # ========================================================

        ttk.Label(
            main,
            text="Environment",
            foreground="#888888"
        ).grid(
            row=14,
            column=1,
            sticky="ew",
            pady=(12, 5)
        )

        # White walls can be combined with any cloud condition
        self.environment_tags = {
            "white_walls": tk.BooleanVar(value=False)
        }

        ttk.Checkbutton(
            main,
            text="White walls",
            variable=self.environment_tags["white_walls"]
        ).grid(
            row=15,
            column=1,
            sticky="w",
            pady=2
        )

        # Clouds / No clouds are mutually exclusive
        self.cloud_condition = tk.StringVar(value="")

        ttk.Radiobutton(
            main,
            text="Clouds",
            variable=self.cloud_condition,
            value="clouds"
        ).grid(
            row=16,
            column=1,
            sticky="w",
            pady=2
        )

        ttk.Radiobutton(
            main,
            text="No clouds",
            variable=self.cloud_condition,
            value="no_clouds"
        ).grid(
            row=17,
            column=1,
            sticky="w",
            pady=2
        )

        # ========================================================
        # PREDICT
        # ========================================================

        self.predict_button = ttk.Button(
            main,
            text="Predict",
            command=self.predict
        )

        self.predict_button.grid(
            row=18,
            column=1,
            sticky="ew",
            pady=(12, 5)
        )

        # ========================================================
        # RESULT
        # ========================================================

        self.result_label = ttk.Label(
            main,
            text="",
            relief="sunken",
            foreground="#00ff00"
        )

        self.result_label.grid(
            row=19,
            column=1,
            sticky="ew",
            pady=5
        )
        # RIGHT CANVAS - Adjusted image preview
        self.canvas_adjusted = tk.Canvas(main, bg="#111111", width=CROP_WIDTH, height=CROP_HEIGHT,
                                        highlightbackground="#333333", highlightthickness=2)
        self.canvas_adjusted.grid(
            row=0,
            column=2,
            rowspan=20,
            padx=10,
            pady=10
        )
        


        # Canvas bindings
        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.end_drag)

    def load_image(self):
        """Load image and enter CROP MODE."""
        path = filedialog.askopenfilename()
        if not path:
            return
        
        self.image_path = path
        pil = Image.open(path).convert("RGB")
        self.original_np = np.array(pil)
        
        # Reset state
        self.mode = "crop_mode"
        self.cropped_np = None
        self.roi = None
        self.crop_region = None
        self.drag_start = None
        
        # Display original (scaled to fit canvas for selection)
        self.display_image_for_crop()
        self.update_mode_label()

    def display_image_for_crop(self):
        """Display original image scaled to canvas for crop selection."""
        self.canvas.delete("all")

        # Compute display size
        img_h, img_w = self.original_np.shape[:2]
        aspect = img_w / img_h

        if aspect > CROP_WIDTH / CROP_HEIGHT:
            display_w = CROP_WIDTH
            display_h = int(CROP_WIDTH / aspect)
        else:
            display_h = CROP_HEIGHT
            display_w = int(CROP_HEIGHT * aspect)

        pil_display = Image.fromarray(self.original_np).resize(
            (display_w, display_h),
            Image.Resampling.LANCZOS
        )

        self.tk_img = ImageTk.PhotoImage(pil_display)

        # CENTER IMAGE AND STORE OFFSET
        self.display_offset_x = (CROP_WIDTH - display_w) // 2
        self.display_offset_y = (CROP_HEIGHT - display_h) // 2

        self.canvas.create_image(
            self.display_offset_x,
            self.display_offset_y,
            anchor="nw",
            image=self.tk_img
        )

        # Scaling factors
        self.canvas_to_original_x = img_w / display_w
        self.canvas_to_original_y = img_h / display_h

    def display_cropped_image(self):
        """Display the 600x500 cropped image at 1:1 scale."""
        pil_img = Image.fromarray(self.cropped_np)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def update_mode_label(self):
        """Update mode indicator label."""
        if self.mode == "crop_mode":
            self.mode_label.config(text="CROP MODE – select 6:5 region")
        elif self.mode == "roi_mode":
            self.mode_label.config(text="ROI MODE – select area for analysis")
        else:
            self.mode_label.config(text="IDLE")

    def reset_crop(self):
        """Return to CROP MODE."""
        if self.image_path:
            self.mode = "crop_mode"
            self.cropped_np = None
            self.roi = None
            self.crop_region = None
            self.drag_start = None
            self.display_image_for_crop()
            self.update_mode_label()

    def start_drag(self, event):
        """Start crop/ROI selection."""
        if self.mode == "crop_mode":
            canvas_x = event.x - self.display_offset_x
            canvas_y = event.y - self.display_offset_y

            if canvas_x < 0 or canvas_y < 0:
                return

            orig_x = int(canvas_x * self.canvas_to_original_x)
            orig_y = int(canvas_y * self.canvas_to_original_y)
            self.drag_start = (orig_x, orig_y)
        elif self.mode == "roi_mode":
            self.drag_start = (event.x, event.y)

    def drag_motion(self, event):
        """Draw preview while dragging."""
        if self.drag_start is None:
            return

        if self.mode == "crop_mode":

            canvas_x = event.x - self.display_offset_x
            canvas_y = event.y - self.display_offset_y

            if canvas_x < 0 or canvas_y < 0:
                return

            orig_x = int(canvas_x * self.canvas_to_original_x)
            orig_y = int(canvas_y * self.canvas_to_original_y)

            x0, y0 = self.drag_start
            dx = orig_x - x0

            # Fixed 6:5 aspect
            width = abs(dx)
            height = int(width / CROP_ASPECT_RATIO)

            if orig_y < y0:
                height = -height

            x1 = x0 + dx
            y1 = y0 + height

            left = min(x0, x1)
            right = max(x0, x1)
            top = min(y0, y1)
            bottom = max(y0, y1)

            img_h, img_w = self.original_np.shape[:2]
            left = max(0, min(left, img_w - 1))
            right = max(0, min(right, img_w))
            top = max(0, min(top, img_h - 1))
            bottom = max(0, min(bottom, img_h))

            self.canvas.delete("crop_rect")

            canvas_left = int(left / self.canvas_to_original_x) + self.display_offset_x
            canvas_top = int(top / self.canvas_to_original_y) + self.display_offset_y
            canvas_right = int(right / self.canvas_to_original_x) + self.display_offset_x
            canvas_bottom = int(bottom / self.canvas_to_original_y) + self.display_offset_y

            self.canvas.create_rectangle(
                canvas_left,
                canvas_top,
                canvas_right,
                canvas_bottom,
                outline="#00ff00",
                width=2,
                tag="crop_rect"
            )

        elif self.mode == "roi_mode":
            self.canvas.delete("roi")
            x0, y0 = self.drag_start
            x1, y1 = event.x, event.y
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="#ff0000", width=2, tag="roi")

    def end_drag(self, event):
        """Finalize crop/ROI selection."""
        if self.drag_start is None:
            return

        if self.mode == "crop_mode":

            canvas_x = event.x - self.display_offset_x
            canvas_y = event.y - self.display_offset_y

            if canvas_x < 0 or canvas_y < 0:
                return

            orig_x = int(canvas_x * self.canvas_to_original_x)
            orig_y = int(canvas_y * self.canvas_to_original_y)

            self.cropped_np, self.crop_region = crop_with_aspect_ratio(
                self.original_np,
                self.drag_start,
                (orig_x, orig_y)
            )

            self.mode = "roi_mode"
            self.roi = None
            self.display_cropped_image()
            self.update_mode_label()
            self.canvas.delete("crop_rect")

        elif self.mode == "roi_mode":
            x0, y0 = self.drag_start
            x1, y1 = event.x, event.y
            self.roi = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))

        self.drag_start = None

    def predict(self):
        """Run prediction pipeline."""
        if self.mode != "roi_mode" or self.roi is None:
            messagebox.showinfo("Error", "Enter ROI MODE and select ROI first.")
            return

        # Extract LAB from ROI in cropped image
        x, y, w, h = self.roi
        if w <= 0 or h <= 0:
            messagebox.showinfo("Error", "Invalid ROI size.")
            return

        roi_img = self.cropped_np[y:y+h, x:x+w]
        lab = rgb_to_lab_mean(roi_img)
        if lab is None:
            messagebox.showinfo("Error", "Could not compute LAB.")
            return

        L, a, b = lab

        # Extract EXIF from original
        iso, exposure, fnumber, hour, minute = extract_exif_data(self.image_path)

        # Get lux values
        try:
            lux_min = float(self.lux_min.get())
            lux_max = float(self.lux_max.get())
        except ValueError:
            messagebox.showinfo("Error", "Enter valid Lux Min and Lux Max.")
            return

        lux_mean = (lux_min + lux_max) / 2

        # Build feature vector
        features = {
            "L": L, "a": a, "b": b,
            "lux_min": lux_min, "lux_max": lux_max, "lux_mean": lux_mean,
            "ISO": iso, "exposure_time": exposure, "f_number": fnumber,
            "hour": hour, "minute": minute
        }

        for tag in self.tags:
            features[f"tag_{tag.replace(' ', '_')}"] = 1 if self.tags[tag].get() else 0

        # Environment features
        features["tag_white_walls"] = (
            1 if self.environment_tags["white_walls"].get() else 0
        )

        features["tag_clouds"] = (
            1 if self.cloud_condition.get() == "clouds" else 0
        )

        features["tag_no_clouds"] = (
            1 if self.cloud_condition.get() == "no_clouds" else 0
        )

        # Prepare and scale data
        df = pd.DataFrame([features])
        df = df.reindex(columns=self.feature_columns, fill_value=0)
        X = self.scaler.transform(df)


        # Predict
        preds = self.model.predict(X)[0]
        wb, r_gain, b_gain, bright, contrast, ev = preds

        # Apply adjustments to cropped image
        adjusted = apply_adjustments(self.cropped_np, r_gain, b_gain, bright, contrast)
        adjusted = apply_exposure_rgb(adjusted, ev)

        self.adjusted_np = adjusted

        # Display adjusted image on RIGHT canvas
        pil_adjusted = Image.fromarray(adjusted)
        adj_h, adj_w = adjusted.shape[:2]
        aspect = adj_w / adj_h
        
        if aspect > CROP_WIDTH / CROP_HEIGHT:
            display_w = CROP_WIDTH
            display_h = int(CROP_WIDTH / aspect)
        else:
            display_h = CROP_HEIGHT
            display_w = int(CROP_HEIGHT * aspect)
        
        pil_display = pil_adjusted.resize((display_w, display_h), Image.Resampling.LANCZOS)
        self.tk_img_adjusted = ImageTk.PhotoImage(pil_display)
        
        self.canvas_adjusted.delete("all")
        self.canvas_adjusted.create_image(CROP_WIDTH // 2, CROP_HEIGHT // 2, anchor="center", image=self.tk_img_adjusted)

        # Display results
        self.result_label.config(
            text=f"WB:{wb:.2f} | R:{r_gain:.2f} | B:{b_gain:.2f} | Br:{bright:.2f} | Ct:{contrast:.2f} | EV:{ev:.2f}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = EOSPredictor(root)
    root.mainloop()