
from __future__ import annotations
import json
import math
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox
from PIL import Image, ImageTk
import numpy as np
from .selection_mask import Mask
from .colorops import (
    rgb_to_hsv,
    luminance,
    recolor_precompute,
    apply_precomputed,
)
from .palette import extract_palette, transfer_map
from .io_utils import pil_open, pil_to_np, np_to_pil

try:  # Pillow < 9 compatibility
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:  # pragma: no cover - fallback for old Pillow
    RESAMPLE_BILINEAR = Image.BILINEAR
    RESAMPLE_NEAREST = Image.NEAREST


class PreviewWorker(threading.Thread):
    def __init__(self, app: 'App'):
        super().__init__(daemon=True)
        self.app = app
        self._queue = queue.Queue()
        self._queue_lock = threading.Lock()
        self.start()

    def schedule(self, payload: dict):
        with self._queue_lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        self._queue.put(payload)

    def flush(self):
        with self._queue_lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    def stop(self):
        self.flush()
        self._queue.put(None)

    def run(self):
        while True:
            payload = self._queue.get()
            if payload is None:
                break
            generation = payload.pop('generation')
            image_id = payload.pop('image_id')
            quality = payload.get('quality', 'full')
            try:
                result = self.app._compute_preview(payload)
            except Exception as exc:  # pragma: no cover - safeguard
                # Surface the error to Tk thread for visibility
                def _raise_error(err=exc):
                    raise err
                self.app.after(0, _raise_error)
                continue
            self.app.after(0, lambda res=result, gen=generation, iid=image_id, q=quality: self.app._on_preview_ready(gen, iid, res, q))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self._prefs_path = os.path.join(os.path.expanduser("~"), ".recolor_rgba_prefs.json")
        self._prefs = self._load_preferences()
        self._tracked_vars = []
        self._preview_job=None
        self._preview_worker=None
        self._preview_generation=0
        self._preview_delay_ms = 0
        self._full_quality_job=None
        self._full_preview_delay_ms = 0
        self._image_serial=0
        self._color_cache=None
        self._draft_np=None
        self._draft_cache=None
        self._draft_scale=1.0
        self._draft_pixel_cap=1_200_000
        self._last_preview_quality="full"
        self._mask_edit_serial = 0
        self._mask_cache_full = None
        self._mask_cache_serial = -1
        self._mask_resampled = {}
        self._mask_last_has_selection = False
        self._tint_full_cache = None
        self._tint_draft_cache = None

        geometry = self._prefs.get("geometry", "1400x860")
        self.title("Recolor RGBA — Pro v1.0")
        self.configure(bg="#0B1020")
        try:
            self.geometry(geometry)
        except tk.TclError:
            self.geometry("1400x860")
        self.minsize(960, 620)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self._tool=self._prefs.get("tool","Brush")
        self._show_mask=bool(self._prefs.get("show_mask", True))
        self.brush_shape=tk.StringVar(value=self._pref_str("brush_shape","Circle")); self._track_var("brush_shape", self.brush_shape)
        self.eraser_shape=tk.StringVar(value=self._pref_str("eraser_shape","Circle")); self._track_var("eraser_shape", self.eraser_shape)
        self.orig_np=None; self.ref_np=None; self.mask=None; self.preview_np=None
        self.preview_dirty=True
        self.zoom = max(0.25, min(6.0, self._pref_float("zoom", 1.0)))
        self.apply_only=tk.BooleanVar(value=self._pref_bool("apply_only", False)); self._track_var("apply_only", self.apply_only)
        self.live_preview=tk.BooleanVar(value=self._pref_bool("live_preview", True)); self._track_var("live_preview", self.live_preview)
        self.status=tk.StringVar(value="Load an image. Left: draw/select; Right: controls. Scroll=zoom, Right-drag=pan.")
        self._rubber_id=None; self._rubber_bbox=None

        self.range_enable=tk.BooleanVar(value=self._pref_bool("range_enable", False)); self._track_var("range_enable", self.range_enable)
        self.range_h_tol=tk.IntVar(value=self._pref_int("range_h_tol", 25)); self._track_var("range_h_tol", self.range_h_tol)
        self.range_s_tol=tk.IntVar(value=self._pref_int("range_s_tol", 20)); self._track_var("range_s_tol", self.range_s_tol)
        self.range_v_tol=tk.IntVar(value=self._pref_int("range_v_tol", 20)); self._track_var("range_v_tol", self.range_v_tol)
        self.range_info=tk.StringVar(value="Affect HSV: –")
        self.range_base=self._read_color_tuple(self._prefs.get("range_base"))
        self.range_hsv=self._read_float_tuple(self._prefs.get("range_hsv"))

        self.exclude_enable=tk.BooleanVar(value=self._pref_bool("exclude_enable", False)); self._track_var("exclude_enable", self.exclude_enable)
        self.exclude_h_tol=tk.IntVar(value=self._pref_int("exclude_h_tol", 25)); self._track_var("exclude_h_tol", self.exclude_h_tol)
        self.exclude_s_tol=tk.IntVar(value=self._pref_int("exclude_s_tol", 20)); self._track_var("exclude_s_tol", self.exclude_s_tol)
        self.exclude_v_tol=tk.IntVar(value=self._pref_int("exclude_v_tol", 20)); self._track_var("exclude_v_tol", self.exclude_v_tol)
        self.exclude_info=tk.StringVar(value="Exclude HSV: –")
        self.exclude_base=self._read_color_tuple(self._prefs.get("exclude_base"))
        self.exclude_hsv=self._read_float_tuple(self._prefs.get("exclude_hsv"))

        self.r=tk.IntVar(value=self._pref_int("target_r", 249)); self._track_var("target_r", self.r)
        self.g=tk.IntVar(value=self._pref_int("target_g", 53)); self._track_var("target_g", self.g)
        self.b=tk.IntVar(value=self._pref_int("target_b", 92)); self._track_var("target_b", self.b)
        self.a=tk.IntVar(value=self._pref_int("target_a", 253)); self._track_var("target_a", self.a)

        self.keep=tk.StringVar(value=self._pref_str("keep", "value")); self._track_var("keep", self.keep)
        self.sat=tk.DoubleVar(value=self._pref_float("sat", 1.0)); self._track_var("sat", self.sat)
        self.alpha_mode=tk.StringVar(value=self._pref_str("alpha_mode", "preserve")); self._track_var("alpha_mode", self.alpha_mode)
        self.tone=tk.DoubleVar(value=self._pref_float("tone", 0.9)); self._track_var("tone", self.tone)
        self.gamma=tk.DoubleVar(value=self._pref_float("gamma", 1.0)); self._track_var("gamma", self.gamma)

        self.brush_size=tk.IntVar(value=self._pref_int("brush_size", 36)); self._track_var("brush_size", self.brush_size)
        self.feather=tk.IntVar(value=self._pref_int("feather", 12)); self._track_var("feather", self.feather)
        self.tolerance=tk.IntVar(value=self._pref_int("tolerance", 24)); self._track_var("tolerance", self.tolerance)
        self.zoom_var = tk.DoubleVar(value=self.zoom); self._track_var("zoom", self.zoom_var)

        self._active_pick=None
        self._build(); self._bind_shortcuts()
        self._preview_worker = PreviewWorker(self)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    @staticmethod
    def _coerce_tuple(value, caster, length):
        if isinstance(value, (list, tuple)) and len(value) == length:
            try:
                return tuple(caster(v) for v in value)
            except (TypeError, ValueError):
                return None
        return None

    def _pref_int(self, key, default):
        try:
            return int(self._prefs.get(key, default))
        except (TypeError, ValueError):
            return default

    def _pref_float(self, key, default):
        try:
            return float(self._prefs.get(key, default))
        except (TypeError, ValueError):
            return default

    def _pref_bool(self, key, default):
        val = self._prefs.get(key, default)
        if isinstance(val, str):
            return val.strip().lower() in {"1","true","yes","on"}
        return bool(val)

    def _pref_str(self, key, default):
        val = self._prefs.get(key, default)
        if val is None:
            return default
        return str(val)

    def _read_color_tuple(self, value):
        tup = self._coerce_tuple(value, int, 3)
        if tup is None:
            return None
        return tuple(max(0, min(255, int(v))) for v in tup)

    def _read_float_tuple(self, value):
        tup = self._coerce_tuple(value, float, 3)
        if tup is None:
            return None
        return tuple(float(v) for v in tup)

    def _track_var(self, key, var):
        self._tracked_vars.append((key, var))

    def _build(self):
        s=ttk.Style(self); s.theme_use("clam")
        s.configure("TFrame", background="#12172A"); s.configure("TLabel", background="#12172A", foreground="#E5E7EB")
        s.configure("TButton", background="#1F243C", foreground="#E5E7EB")
        paned = ttk.Panedwindow(self, orient="horizontal"); paned.pack(fill="both", expand=True, padx=8, pady=8)
        left = ttk.Frame(paned); right_shell = ttk.Frame(paned)
        paned.add(left, weight=3); paned.add(right_shell, weight=2)

        # LEFT
        left.columnconfigure(0, weight=1)
        cframe = ttk.Frame(left); cframe.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(cframe, width=820, height=520, bg="#0E1224", highlightthickness=0, cursor="tcross")
        hbar = tk.Scrollbar(cframe, orient="horizontal"); vbar = tk.Scrollbar(cframe, orient="vertical")
        hbar.config(command=self.canvas.xview); vbar.config(command=self.canvas.yview)
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0,column=0,sticky="nsew"); vbar.grid(row=0,column=1,sticky="ns"); hbar.grid(row=1,column=0,sticky="ew")
        cframe.grid_columnconfigure(0, weight=1); cframe.grid_rowconfigure(0, weight=1)

        bar=ttk.Frame(left); bar.pack(fill="x",pady=4)
        for n in ["Brush","Eraser","Rect","Ellipse","Wand","Invert","Clear","Undo","Redo","Show/Hide"]:
            ttk.Button(bar,text=n,command=lambda x=n:self._tool_action(x)).pack(side="left",padx=3)

        ctr=ttk.Frame(left); ctr.pack(fill="x",pady=6)
        ttk.Label(ctr,text="Size").pack(side="left"); tk.Scale(ctr,from_=2,to=200,variable=self.brush_size,orient="horizontal",length=150,command=lambda e:self._refresh()).pack(side="left",padx=6)
        ttk.Label(ctr,text="Feather").pack(side="left"); tk.Scale(ctr,from_=0,to=100,variable=self.feather,orient="horizontal",length=150,command=lambda e:self._refresh()).pack(side="left",padx=6)
        ttk.Label(ctr,text="Tolerance").pack(side="left"); tk.Scale(ctr,from_=0,to=120,variable=self.tolerance,orient="horizontal",length=150).pack(side="left",padx=6)
        ttk.Label(ctr,text="Zoom ×").pack(side="left")
        tk.Scale(ctr,from_=0.25,to=6.0,resolution=0.05,variable=self.zoom_var,orient="horizontal",length=220,command=self.on_zoom).pack(side="left",padx=6)

        shape_row=ttk.Frame(left); shape_row.pack(fill="x",pady=4)
        ttk.Label(shape_row,text="Brush shape").pack(side="left")
        ttk.Combobox(shape_row, values=["Circle","Square"], textvariable=self.brush_shape, width=9, state="readonly").pack(side="left", padx=6)
        ttk.Label(shape_row,text="Eraser shape").pack(side="left", padx=(12,0))
        ttk.Combobox(shape_row, values=["Circle","Square"], textvariable=self.eraser_shape, width=9, state="readonly").pack(side="left", padx=6)

        lf=ttk.Frame(left); lf.pack(fill="x")
        ttk.Button(lf,text="Load Original…",command=self.load_original).pack(side="left",padx=3)
        ttk.Button(lf,text="Load Reference…",command=self.load_reference).pack(side="left",padx=3)
        ttk.Button(lf,text="Save PNG…",command=self.save_png).pack(side="left",padx=3)
        ttk.Checkbutton(lf, text="Apply to selection only", variable=self.apply_only, command=self.preview).pack(side="left", padx=12)
        ttk.Checkbutton(lf, text="Live preview", variable=self.live_preview, command=self._on_live_preview_toggle).pack(side="left", padx=12)

        ttk.Label(self, textvariable=self.status).pack(fill="x", side="bottom")

        # RIGHT (scrollable)
        right_canvas = tk.Canvas(right_shell, bg="#12172A", highlightthickness=0)
        right_scroll = ttk.Scrollbar(right_shell, orient="vertical", command=right_canvas.yview)
        right_canvas.configure(yscrollcommand=right_scroll.set)
        right_canvas.pack(side="left", fill="both", expand=True)
        right_scroll.pack(side="right", fill="y")
        right = ttk.Frame(right_canvas)
        right_window = right_canvas.create_window((0,0), window=right, anchor="nw")

        def _sync_right(event):
            right_canvas.configure(scrollregion=right_canvas.bbox("all"))
            right_canvas.itemconfigure(right_window, width=event.width)

        right.bind("<Configure>", _sync_right)
        right_canvas.bind("<Configure>", lambda e: right_canvas.itemconfigure(right_window, width=e.width))

        def _scroll_right(event):
            delta = 0
            if getattr(event, 'delta', 0):
                delta = -1 if event.delta > 0 else 1
            elif getattr(event, 'num', None) in (4, 5):
                delta = -1 if event.num == 4 else 1
            if delta:
                right_canvas.yview_scroll(delta, "units")
            return "break"

        right_canvas.bind("<MouseWheel>", _scroll_right)
        right_canvas.bind("<Button-4>", _scroll_right)
        right_canvas.bind("<Button-5>", _scroll_right)
        right.bind("<MouseWheel>", _scroll_right)

        cf=ttk.Labelframe(right,text="Target Color"); cf.pack(fill="x",pady=6)
        for (lbl,var) in [("R",self.r),("G",self.g),("B",self.b),("A",self.a)]:
            ttk.Label(cf,text=lbl).pack(side="left"); tk.Scale(cf,from_=0,to=255,variable=var,orient="horizontal",length=160,command=lambda _evt: self._on_recolor_param_change()).pack(side="left",padx=6)
        ttk.Button(cf,text="Pick…",command=self.pick_color).pack(side="left",padx=6)

        tf=ttk.Labelframe(right,text="Tone / Hue"); tf.pack(fill="x",pady=6)
        ttk.Label(tf,text="Preserve").pack(side="left");
        keep_combo = ttk.Combobox(tf,values=["value","luminance"],textvariable=self.keep,width=10)
        keep_combo.pack(side="left",padx=6)
        keep_combo.bind("<<ComboboxSelected>>", self._on_recolor_param_change_event)
        ttk.Label(tf,text="Sat ×").pack(side="left"); tk.Scale(tf,from_=0,to=2,resolution=0.01,variable=self.sat,orient="horizontal",length=180,command=lambda _evt: self._on_recolor_param_change()).pack(side="left",padx=6)
        ttk.Label(tf,text="Alpha").pack(side="left");
        alpha_combo = ttk.Combobox(tf,values=["preserve","multiply"],textvariable=self.alpha_mode,width=10)
        alpha_combo.pack(side="left",padx=6)
        alpha_combo.bind("<<ComboboxSelected>>", self._on_recolor_param_change_event)
        ttk.Label(tf,text="Tone blend").pack(side="left"); tk.Scale(tf,from_=0,to=1,resolution=0.01,variable=self.tone,orient="horizontal",length=180,command=lambda _evt: self._on_recolor_param_change()).pack(side="left",padx=6)
        ttk.Label(tf,text="Shading γ").pack(side="left"); tk.Scale(tf,from_=0.4,to=2,resolution=0.01,variable=self.gamma,orient="horizontal",length=180,command=lambda _evt: self._on_recolor_param_change()).pack(side="left",padx=6)

        rf=ttk.Labelframe(right,text="Color Ranges"); rf.pack(fill="x",pady=6)
        self._build_threshold_section(rf, kind="include", title="Affect range")
        ttk.Separator(rf, orient="horizontal").pack(fill="x", padx=6, pady=4)
        self._build_threshold_section(rf, kind="exclude", title="Exclude range")

        row = ttk.Frame(right); row.pack(fill="x",pady=6)
        self.thumb_orig = tk.Canvas(row, width=260, height=140, bg="#0E1224", highlightthickness=0); self.thumb_orig.pack(side="left", padx=6)
        refblock = ttk.Frame(row); refblock.pack(side="left", padx=6)
        self.thumb_ref  = tk.Canvas(refblock, width=260, height=140, bg="#0E1224", highlightthickness=0); self.thumb_ref.pack()
        self.pal_frame = ttk.Frame(refblock); self.pal_frame.pack(fill="x", pady=4)
        self.thumb_res  = tk.Canvas(row, width=260, height=140, bg="#0E1224", highlightthickness=0); self.thumb_res.pack(side="left", padx=6)
        pb = ttk.Frame(right); pb.pack(fill="x", pady=2)
        ttk.Button(pb, text="Extract palette (top4)", command=self.extract_palette).pack(side="left", padx=6)
        ttk.Button(pb, text="Transfer 4→4", command=self.transfer_4x4).pack(side="left", padx=6)

        # bindings
        self.canvas.bind("<Button-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel); self.canvas.bind("<Button-4>", self.on_mousewheel); self.canvas.bind("<Button-5>", self.on_mousewheel)
        self.canvas.bind("<Button-3>", self.on_pan_start); self.canvas.bind("<B3-Motion>", self.on_pan_move)
        self.canvas.bind("<Button-2>", self.on_pan_start); self.canvas.bind("<B2-Motion>", self.on_pan_move)

        self._update_canvas_cursor()

    def _bind_shortcuts(self):
        for key,tool in [("b","Brush"),("e","Eraser"),("r","Rect"),("o","Ellipse"),("w","Wand"),("i","Invert"),("c","Clear"),("u","Undo"),("y","Redo"),("m","Show/Hide")]:
            self.bind(key, lambda e, t=tool: self._tool_action(t))
        self.bind("<Control-o>", lambda e: self.load_original())
        self.bind("<Control-Shift-O>", lambda e: self.load_reference())
        self.bind("<Control-s>", lambda e: self.save_png())
        self.bind("+", lambda e: self._zoom_step(+0.1)); self.bind("-", lambda e: self._zoom_step(-0.1)); self.bind("0", lambda e: self._reset_zoom())

    def _build_threshold_section(self, parent, kind, title):
        frame = ttk.Frame(parent); frame.pack(fill="x", pady=2)
        enable = self.range_enable if kind=="include" else self.exclude_enable
        h_var = self.range_h_tol if kind=="include" else self.exclude_h_tol
        s_var = self.range_s_tol if kind=="include" else self.exclude_s_tol
        v_var = self.range_v_tol if kind=="include" else self.exclude_v_tol
        info_var = self.range_info if kind=="include" else self.exclude_info

        header = ttk.Frame(frame); header.pack(fill="x", pady=2)
        ttk.Checkbutton(header, text=title, variable=enable, command=lambda k=kind: self._on_threshold_toggle(k)).pack(side="left", padx=(6,4))
        swatch = tk.Canvas(header, width=34, height=34, highlightthickness=1, highlightbackground="#ffffff", bg="#0E1224")
        swatch.pack(side="left", padx=4)
        if kind=="include":
            self.range_swatch = swatch
        else:
            self.exclude_swatch = swatch
        ttk.Button(header, text="Pick", command=lambda k=kind: self._begin_threshold_pick(k)).pack(side="left", padx=3)
        ttk.Button(header, text="Choose…", command=lambda k=kind: self._choose_threshold_color(k)).pack(side="left", padx=3)

        ttk.Label(frame, textvariable=info_var).pack(anchor="w", padx=10)

        sliders = ttk.Frame(frame); sliders.pack(fill="x", pady=2)
        ttk.Label(sliders, text="Hue ±°").grid(row=0, column=0, sticky="w", padx=4)
        tk.Scale(sliders, from_=0, to=180, variable=h_var, orient="horizontal", length=220,
                 command=lambda _evt, k=kind: self._on_threshold_change(k)).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Label(sliders, text="Sat ±%" ).grid(row=1, column=0, sticky="w", padx=4)
        tk.Scale(sliders, from_=0, to=100, variable=s_var, orient="horizontal", length=220,
                 command=lambda _evt, k=kind: self._on_threshold_change(k)).grid(row=1, column=1, sticky="we", padx=4)
        ttk.Label(sliders, text="Val ±%" ).grid(row=2, column=0, sticky="w", padx=4)
        tk.Scale(sliders, from_=0, to=100, variable=v_var, orient="horizontal", length=220,
                 command=lambda _evt, k=kind: self._on_threshold_change(k)).grid(row=2, column=1, sticky="we", padx=4)
        sliders.grid_columnconfigure(1, weight=1)

        self._update_threshold_widgets(kind)

    # helpers
    def _zoom_step(self, dz):
        z = max(0.25, min(6.0, self.zoom + dz)); self.zoom_var.set(z); self.zoom = z; self._refresh()
    def _reset_zoom(self):
        if self.orig_np is None: return
        H, W = self.orig_np.shape[0], self.orig_np.shape[1]
        self.update_idletasks()
        canvas_w = max(1, self.canvas.winfo_width()) if hasattr(self, 'canvas') else 820
        canvas_h = max(1, self.canvas.winfo_height()) if hasattr(self, 'canvas') else 520
        z = min(canvas_w/float(max(1,W)), canvas_h/float(max(1,H)), 1.0)
        z = max(0.25, z)
        self.zoom_var.set(z); self.zoom = z; self._refresh()
    def _update_canvas_cursor(self):
        if not hasattr(self, "canvas"):
            return
        if self._active_pick is not None:
            cursor = "dotbox"
        elif self._tool in ("Brush","Eraser"):
            cursor = "spraycan"
        else:
            cursor = "tcross"
        self.canvas.configure(cursor=cursor)
    def on_zoom(self, *_): self.zoom = float(self.zoom_var.get()); self._refresh()
    def on_mousewheel(self, e): delta = 1 if getattr(e,"delta",0)>0 or getattr(e,"num",0)==4 else -1; self._zoom_step(delta*0.1)
    def on_pan_start(self, e): self.canvas.scan_mark(e.x, e.y)
    def on_pan_move(self, e): self.canvas.scan_dragto(e.x, e.y, gain=1)
    def _event_to_img_xy(self, e): return int(self.canvas.canvasx(e.x) / self.zoom), int(self.canvas.canvasy(e.y) / self.zoom)

    # I/O
    def load_original(self):
        p=filedialog.askopenfilename(filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.webp")])
        if not p: return
        im=pil_open(p); npimg=pil_to_np(im); self.orig_np=npimg; H,W,_=npimg.shape
        self.mask=Mask(H,W)
        self._mark_mask_dirty()
        self._color_cache = self._build_color_cache(npimg)
        self._setup_preview_buffers(npimg)
        self._invalidate_tint_cache()
        self._last_preview_quality = "full"
        self._image_serial += 1
        if self._preview_worker is not None:
            self._preview_worker.flush()
        if self._full_quality_job is not None:
            try:
                self.after_cancel(self._full_quality_job)
            finally:
                self._full_quality_job=None
        self.update_idletasks()
        canvas_w = max(1, self.canvas.winfo_width()) if hasattr(self, 'canvas') else 820
        canvas_h = max(1, self.canvas.winfo_height()) if hasattr(self, 'canvas') else 520
        fit_zoom = min(canvas_w/float(max(1,W)), canvas_h/float(max(1,H)), 1.0)
        self.zoom = max(0.25, fit_zoom)
        self.zoom_var.set(self.zoom)
        self._update_threshold_widgets("include")
        self._update_threshold_widgets("exclude")
        self._active_pick=None; self._update_canvas_cursor()
        self._draw_thumb(self.thumb_orig, self.orig_np)  # "ANTES" fixo
        self._cancel_preview_job()
        self.preview_np=None  # força render original até a primeira preview
        self.preview_dirty=True
        self._render_left()
        self.preview(immediate=True)

    def load_reference(self):
        p=filedialog.askopenfilename(filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.webp")])
        if not p: return
        self.ref_np=pil_to_np(pil_open(p)); self._draw_thumb(self.thumb_ref, self.ref_np); self._clear_palette_swatches()

    def save_png(self):
        if self.preview_np is None or self.preview_dirty: self.preview(immediate=True, force_full=True)
        p=filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")])
        if not p: return
        np_to_pil(self.preview_np).save(p)

    # thumbnails / render
    def _build_color_cache(self, npimg):
        if npimg is None:
            return None
        src = npimg.astype(np.float32) / 255.0
        r = src[...,0]
        g = src[...,1]
        b = src[...,2]
        a = src[...,3]
        h,s,v = rgb_to_hsv(r,g,b)
        Yo = luminance(r,g,b)
        return {
            'src': src,
            'r': r,
            'g': g,
            'b': b,
            'a': a,
            'h': h,
            's': s,
            'v': v,
            'Yo': Yo,
        }

    def _setup_preview_buffers(self, npimg):
        self._draft_np=None
        self._draft_cache=None
        self._draft_scale=1.0
        self._tint_draft_cache = None
        if npimg is None:
            return
        H,W,_ = npimg.shape
        total = H * W
        if total <= self._draft_pixel_cap:
            return
        scale = math.sqrt(self._draft_pixel_cap / float(total))
        scale = max(0.18, min(scale, 1.0))
        new_w = max(1, int(W * scale))
        new_h = max(1, int(H * scale))
        if new_w == W and new_h == H:
            return
        down = Image.fromarray(npimg, mode="RGBA").resize((new_w, new_h), RESAMPLE_BILINEAR)
        self._draft_np = np.array(down, dtype=np.uint8)
        self._draft_cache = self._build_color_cache(self._draft_np)
        self._draft_scale = scale

    def _mark_mask_dirty(self):
        self._mask_edit_serial += 1
        self._mask_cache_full = None
        self._mask_cache_serial = -1
        self._mask_resampled.clear()
        self._mask_last_has_selection = False

    def _mask_empty_for_shape(self, shape):
        shape_key = tuple(int(v) for v in shape)
        key = ("empty", shape_key)
        cached = self._mask_resampled.get(key)
        if cached is not None and cached[0] == self._mask_edit_serial:
            return cached[1]
        arr = np.zeros(shape_key, dtype=np.float32)
        self._mask_resampled[key] = (self._mask_edit_serial, arr)
        return arr

    def _mask_full_array(self):
        if self.mask is None or self.orig_np is None:
            return None
        has_selection = self.mask.any_selected()
        self._mask_last_has_selection = has_selection
        base_shape = self.mask.alpha.shape
        if not has_selection:
            empty = self._mask_empty_for_shape(base_shape)
            self._mask_cache_full = empty
            self._mask_cache_serial = self._mask_edit_serial
            return empty
        if self._mask_cache_full is None or self._mask_cache_serial != self._mask_edit_serial:
            self._mask_cache_full = self.mask.alpha.astype(np.float32, copy=True)
            self._mask_cache_serial = self._mask_edit_serial
        return self._mask_cache_full

    def _mask_for_shape(self, target_shape=None):
        if self.mask is None or self.orig_np is None or not self.apply_only.get():
            return None
        base = self._mask_full_array()
        if base is None:
            return None
        if target_shape is None:
            return base
        target_key = tuple(int(v) for v in target_shape)
        if base.shape == target_key:
            return base
        key = ("resampled", target_key)
        cached = self._mask_resampled.get(key)
        if cached is not None and cached[0] == self._mask_cache_serial:
            return cached[1]
        if not self._mask_last_has_selection:
            resampled = self._mask_empty_for_shape(target_key)
        else:
            resampled = self._resample_mask_to(base, target_key)
        self._mask_resampled[key] = (self._mask_cache_serial, resampled)
        return resampled

    def _invalidate_tint_cache(self):
        self._tint_full_cache = None
        self._tint_draft_cache = None

    def _current_recolor_signature(self):
        def _round(val):
            return round(float(val), 6)

        include_base = tuple(int(v) for v in self.range_base) if self.range_base is not None else None
        include_hsv = (
            tuple(_round(v) for v in self.range_hsv) if self.range_hsv is not None else None
        )
        exclude_base = tuple(int(v) for v in self.exclude_base) if self.exclude_base is not None else None
        exclude_hsv = (
            tuple(_round(v) for v in self.exclude_hsv) if self.exclude_hsv is not None else None
        )
        return (
            int(self.r.get()),
            int(self.g.get()),
            int(self.b.get()),
            int(self.a.get()),
            self.keep.get(),
            _round(self.sat.get()),
            self.alpha_mode.get(),
            _round(self.tone.get()),
            _round(self.gamma.get()),
            bool(self.range_enable.get()),
            include_base,
            include_hsv,
            int(self.range_h_tol.get()),
            int(self.range_s_tol.get()),
            int(self.range_v_tol.get()),
            bool(self.exclude_enable.get()),
            exclude_base,
            exclude_hsv,
            int(self.exclude_h_tol.get()),
            int(self.exclude_s_tol.get()),
            int(self.exclude_v_tol.get()),
        )

    def _ensure_precomputed(self, image_np, cache, quality, color_threshold, exclude_threshold):
        signature = self._current_recolor_signature()
        attr = '_tint_full_cache' if quality == 'full' else '_tint_draft_cache'
        store = getattr(self, attr)
        if (
            store is None
            or store.get('image_id') != self._image_serial
            or store.get('signature') != signature
        ):
            src, tinted, gate = recolor_precompute(
                image_np,
                (self.r.get(), self.g.get(), self.b.get(), self.a.get()),
                keep=self.keep.get(),
                saturation_scale=float(self.sat.get()),
                alpha_mode=self.alpha_mode.get(),
                tone_blend=float(self.tone.get()),
                shade_gamma=float(self.gamma.get()),
                color_threshold=color_threshold,
                exclude_threshold=exclude_threshold,
                cache=cache,
            )
            store = {
                'image_id': self._image_serial,
                'signature': signature,
                'src': src,
                'tinted': tinted,
                'gate': gate,
            }
            setattr(self, attr, store)
        return store

    def _resample_mask_to(self, mask, target_shape):
        if mask is None:
            return None
        target_h, target_w = target_shape
        if mask.shape == (target_h, target_w):
            return mask
        pil_mask = Image.fromarray((np.clip(mask, 0, 1) * 255.0 + 0.5).astype(np.uint8), mode="L")
        resized = pil_mask.resize((target_w, target_h), RESAMPLE_BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        return arr

    def _upsample_preview(self, npimg):
        if npimg is None or self.orig_np is None:
            return npimg
        H,W,_ = self.orig_np.shape
        if npimg.shape[0] == H and npimg.shape[1] == W:
            return npimg
        up = Image.fromarray(npimg, mode="RGBA").resize((W, H), RESAMPLE_BILINEAR)
        return np.array(up, dtype=np.uint8)

    def _queue_full_quality(self, generation):
        if self._draft_np is None:
            return
        if self._full_quality_job is not None:
            try:
                self.after_cancel(self._full_quality_job)
            except tk.TclError:
                pass
        self._full_quality_job = None
        delay = max(0, self._full_preview_delay_ms)
        if delay <= 0:
            self._full_quality_job = self.after(
                0,
                lambda g=generation: self._run_preview(generation=g, quality="full"),
            )
        else:
            self._full_quality_job = self.after(
                delay,
                lambda g=generation: self._run_preview(generation=g, quality="full"),
            )

    def _draw_thumb(self, canvas, npimg):
        if npimg is None: return
        canvas.delete("all")
        im = Image.fromarray(npimg, mode="RGBA"); W,H = im.size; w,h = int(canvas['width']), int(canvas['height'])
        scale = min(w/W, h/H, 1.0); disp = im.resize((int(W*scale), int(H*scale)), RESAMPLE_NEAREST)
        tkimg = ImageTk.PhotoImage(disp); canvas.image = tkimg; canvas.create_image(0,0,image=tkimg,anchor="nw")

    def _render_left(self, overlay_mask=None):
        if self.orig_np is None: return
        # >>> use PREVIEW image if we have it (live recolor on big canvas)
        use_preview = self.preview_np is not None and self.live_preview.get()
        base_np = self.preview_np if use_preview else self.orig_np
        base = Image.fromarray(base_np, mode="RGBA")

        # overlay selection (current + temporary)
        if getattr(self,"_show_mask",True) and self.mask is not None:
            a = self.mask.as_uint8_alpha() if self.mask.any_selected() else None
            if overlay_mask is not None:
                a_tmp = (np.clip(overlay_mask,0,1)*255.0+0.5).astype(np.uint8)
                a = a_tmp if a is None else np.maximum(a, a_tmp)
            if a is not None and (a>0).any():
                overlay = np.dstack([np.full_like(a,255), np.zeros_like(a), np.zeros_like(a), (a*220)//255])
                base = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))

        W,H = base.size; disp = base.resize((int(W*self.zoom), int(H*self.zoom)), Image.NEAREST)
        self._tk = ImageTk.PhotoImage(disp); self.canvas.config(scrollregion=(0,0,disp.size[0],disp.size[1]))
        self.canvas.delete("all"); self.canvas.create_image(0,0,image=self._tk,anchor="nw")

        # keep rubberband on top
        if self._rubber_id is not None and self._rubber_bbox is not None:
            x0,y0,x1,y1 = self._rubber_bbox
            rx0,ry0,rx1,ry1 = [v*self.zoom for v in (x0,y0,x1,y1)]
            self._rubber_id = self.canvas.create_rectangle(rx0,ry0,rx1,ry1, outline="#ffffff", dash=(4,3))

    def _refresh(self): self._render_left()

    # tools
    def _tool_action(self, name):
        if name in ("Brush","Eraser","Rect","Ellipse","Wand"):
            self._tool=name
            self._update_canvas_cursor()
            self.status.set(f"Tool: {name}. Left drag. Right=pan. Scroll=zoom.")
            return
        if self.mask is None: return
        changed = False
        if name=="Invert":
            self.mask.invert(); changed = True
        elif name=="Clear":
            self.mask.clear(); changed = True
        elif name=="Undo":
            self.mask.undo(); changed = True
        elif name=="Redo":
            self.mask.redo(); changed = True
        elif name=="Show/Hide":
            self._show_mask=not getattr(self,"_show_mask",True)
        if changed:
            self._mark_mask_dirty()
        # After state changes, update preview and left
        self._update_canvas_cursor()
        self._render_live_preview()
        self.preview()

    def on_down(self,e):
        if self.orig_np is None or self.mask is None: return
        ix,iy = self._event_to_img_xy(e); self._x0,self._y0 = ix,iy
        if self._active_pick is not None:
            kind=self._active_pick
            self._apply_threshold_sample(kind, ix, iy)
            return
        if self._tool in ("Brush","Eraser"):
            sign=+1.0 if self._tool=="Brush" else -1.0
            shape = self.brush_shape.get() if self._tool=="Brush" else self.eraser_shape.get()
            self.mask.brush(ix,iy,self.brush_size.get(),self.feather.get(),sign=sign, shape=shape, snapshot=True)
            self._mark_mask_dirty()
            self._render_live_preview()
        elif self._tool=="Wand":
            self.mask.magic_wand(self.orig_np, ix, iy, tolerance=self.tolerance.get(), sign=+1.0)
            self._mark_mask_dirty()
            self._render_live_preview()
        elif self._tool in ("Rect","Ellipse"):
            self._rubber_bbox=(ix,iy,ix,iy); self._rubber_id=None
            self._render_live_preview(temp_shape=True)

    def on_drag(self,e):
        if self.orig_np is None or self.mask is None: return
        ix,iy = self._event_to_img_xy(e)
        if self._tool in ("Brush","Eraser"):
            sign=+1.0 if self._tool=="Brush" else -1.0
            shape = self.brush_shape.get() if self._tool=="Brush" else self.eraser_shape.get()
            self.mask.brush(ix,iy,self.brush_size.get(),self.feather.get(),sign=sign, shape=shape, snapshot=False)
            self._mark_mask_dirty()
            self._render_live_preview()
        elif self._tool in ("Rect","Ellipse") and self._rubber_bbox is not None:
            self._rubber_bbox=(self._x0,self._y0,ix,iy)
            self._render_live_preview(temp_shape=True)

    def on_up(self,e):
        if self.orig_np is None or self.mask is None: return
        ix,iy = self._event_to_img_xy(e)
        if self._tool=="Rect":
            self.mask.rect(self._x0,self._y0,ix,iy,self.feather.get(),sign=+1.0); self._mark_mask_dirty()
        elif self._tool=="Ellipse":
            self.mask.ellipse(self._x0,self._y0,ix,iy,self.feather.get(),sign=+1.0); self._mark_mask_dirty()
        if self._rubber_id is not None:
            self.canvas.delete(self._rubber_id); self._rubber_id=None; self._rubber_bbox=None
        self._render_live_preview()  # final
        self.preview(immediate=True)

    # live preview composer
    def _build_temp_mask(self):
        if self._tool=="Rect" and self._rubber_bbox is not None:
            x0,y0,x1,y1 = self._rubber_bbox; return self.mask._rect_alpha(x0,y0,x1,y1,self.feather.get())
        if self._tool=="Ellipse" and self._rubber_bbox is not None:
            x0,y0,x1,y1 = self._rubber_bbox; return self.mask._ellipse_alpha(x0,y0,x1,y1,self.feather.get())
        return None

    def _render_live_preview(self, temp_shape=False):
        temp = self._build_temp_mask() if temp_shape else None
        self.preview()
        self._render_left(overlay_mask=temp)

    def _on_recolor_param_change(self, *_):
        self._invalidate_tint_cache()
        self.preview()

    def _on_recolor_param_change_event(self, *_):
        self._on_recolor_param_change()

    # color + palette
    def pick_color(self):
        rgb,_ = colorchooser.askcolor()
        if rgb:
            r,g,b = map(int,rgb)
            self.r.set(r); self.g.set(g); self.b.set(b)
            self._invalidate_tint_cache()
            self.preview()

    def _threshold_title(self, kind):
        return "affect" if kind=="include" else "exclude"

    def _begin_threshold_pick(self, kind):
        if self.orig_np is None:
            messagebox.showinfo("Pick color","Load an image before sampling colors.")
            return
        self._active_pick=kind
        action=self._threshold_title(kind)
        self.status.set(f"Click on the image to sample the {action} color.")
        self._update_canvas_cursor()

    def _choose_threshold_color(self, kind):
        rgb,_ = colorchooser.askcolor()
        if rgb:
            col = tuple(map(int,rgb))
            self._active_pick=None
            self._update_canvas_cursor()
            self._set_threshold_base(kind, col)
            action=self._threshold_title(kind)
            self.status.set(f"Selected #{col[0]:02x}{col[1]:02x}{col[2]:02x} to {action}.")

    def _apply_threshold_sample(self, kind, ix, iy):
        self._active_pick=None
        self._update_canvas_cursor()
        if self.orig_np is None:
            return
        H,W,_ = self.orig_np.shape
        if not (0 <= ix < W and 0 <= iy < H):
            self.status.set("Click inside the image to sample.")
            return
        rgb = tuple(int(v) for v in self.orig_np[iy,ix,:3])
        self._set_threshold_base(kind, rgb)
        action=self._threshold_title(kind)
        self.status.set(f"Sampled #{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x} to {action}.")

    def _set_threshold_base(self, kind, rgb):
        r,g,b=[v/255.0 for v in rgb]
        h,s,v = rgb_to_hsv(np.array([r]),np.array([g]),np.array([b]))
        hsv_tuple = (float(h[0]), float(s[0]), float(v[0]))
        if kind=="include":
            self.range_base=rgb; self.range_hsv=hsv_tuple
        else:
            self.exclude_base=rgb; self.exclude_hsv=hsv_tuple
        self._update_threshold_widgets(kind)
        self._invalidate_tint_cache()
        self.preview()

    def _update_threshold_widgets(self, kind):
        swatch = getattr(self, "range_swatch" if kind=="include" else "exclude_swatch", None)
        base = self.range_base if kind=="include" else self.exclude_base
        hsv = self.range_hsv if kind=="include" else self.exclude_hsv
        info_var = self.range_info if kind=="include" else self.exclude_info
        label = "Affect" if kind=="include" else "Exclude"
        if swatch is not None:
            swatch.delete("all")
            if base is None:
                swatch.create_rectangle(0,0,34,34, fill="#0E1224", outline="")
            else:
                r,g,b=base
                swatch.create_rectangle(0,0,34,34, fill=f"#{r:02x}{g:02x}{b:02x}", outline="")
        if hsv is None:
            info_var.set(f"{label} HSV: –")
        else:
            hdeg=int(hsv[0]*360.0+0.5)
            sper=int(hsv[1]*100.0+0.5)
            vper=int(hsv[2]*100.0+0.5)
            info_var.set(f"{label} HSV: {hdeg}° / {sper}% / {vper}%")

    def _on_threshold_change(self, kind):
        self._invalidate_tint_cache()
        self.preview()

    def _on_threshold_toggle(self, kind):
        self._invalidate_tint_cache()
        self.preview()
        self._update_threshold_widgets(kind)

    def _clear_palette_swatches(self):
        for child in list(self.pal_frame.pack_slaves()): child.destroy()

    def _add_swatch(self, rgba, ratio):
        sw = tk.Canvas(self.pal_frame, width=40, height=20, highlightthickness=1, highlightbackground="#fff"); sw.pack(side="left", padx=4)
        sw.create_rectangle(0,0,40,20, fill=f"#{rgba[0]:02x}{rgba[1]:02x}{rgba[2]:02x}", outline="")
        def set_color(_evt=None, col=rgba):
            self.r.set(int(col[0])); self.g.set(int(col[1])); self.b.set(int(col[2])); self.a.set(int(col[3]));
            self._invalidate_tint_cache()
            self.preview()
        sw.bind("<Button-1>", set_color)

    def extract_palette(self):
        if self.ref_np is None: messagebox.showwarning("Palette","Load reference first"); return
        self._clear_palette_swatches(); pal = extract_palette(self.ref_np,4)
        for rgba, ratio in pal: self._add_swatch(rgba, ratio)

    def transfer_4x4(self):
        if self.ref_np is None or self.orig_np is None: return
        palO=extract_palette(self.orig_np,4); palR=extract_palette(self.ref_np,4)
        if not palO or not palR: return
        mapping=transfer_map(palO,palR); target=palR[mapping[0]][0]
        self.r.set(int(target[0])); self.g.set(int(target[1])); self.b.set(int(target[2])); self.a.set(int(target[3]));
        self._invalidate_tint_cache()
        self.preview()

    # preview
    def preview(self,*_, immediate=False, force_full=False):
        if self.orig_np is None:
            return
        self._invalidate_preview(immediate=immediate, force_full=force_full)

    def _cancel_preview_job(self):
        if self._preview_job is not None:
            try:
                self.after_cancel(self._preview_job)
            finally:
                self._preview_job=None
        if self._full_quality_job is not None:
            try:
                self.after_cancel(self._full_quality_job)
            finally:
                self._full_quality_job=None

    def _invalidate_preview(self, immediate=False, force_full=False):
        self.preview_dirty=True
        self._preview_generation += 1
        generation = self._preview_generation
        self._cancel_preview_job()
        if immediate:
            if force_full or self._draft_np is None:
                quality = "full"
            else:
                quality = "draft"
            self._run_preview(generation=generation, blocking=True, quality=quality)
            return
        preferred_quality = "draft" if self._draft_np is not None else "full"
        if force_full:
            preferred_quality = "full"
        self._run_preview(generation=generation, blocking=True, quality=preferred_quality)

    def _current_threshold(self, kind="include"):
        enable = self.range_enable if kind=="include" else self.exclude_enable
        base = self.range_base if kind=="include" else self.exclude_base
        hsv = self.range_hsv if kind=="include" else self.exclude_hsv
        h_var = self.range_h_tol if kind=="include" else self.exclude_h_tol
        s_var = self.range_s_tol if kind=="include" else self.exclude_s_tol
        v_var = self.range_v_tol if kind=="include" else self.exclude_v_tol
        if not enable.get() or base is None:
            return None
        return {
            "base": base,
            "base_hsv": hsv,
            "h_tolerance": h_var.get(),
            "s_tolerance": s_var.get()/100.0,
            "v_tolerance": v_var.get()/100.0,
        }

    def _run_preview(self, generation, blocking=False, quality="auto"):
        self._preview_job=None
        if quality == "full":
            requested_quality = "full"
        elif quality == "draft" and self._draft_np is not None and self._draft_cache is not None:
            requested_quality = "draft"
        elif quality == "auto" and self._draft_np is not None and self._draft_cache is not None:
            requested_quality = "draft"
        else:
            requested_quality = "full"
        if requested_quality == "full":
            image_np = self.orig_np
            cache = self._color_cache
            mask_shape = None
            self._full_quality_job = None
        else:
            image_np = self._draft_np
            cache = self._draft_cache
            mask_shape = image_np.shape[:2]
            self._queue_full_quality(generation)
        if self.orig_np is None or generation != self._preview_generation:
            return
        mask_array = self._mask_for_shape(mask_shape)
        color_threshold = self._current_threshold("include")
        exclude_threshold = self._current_threshold("exclude")
        precomputed = self._ensure_precomputed(
            image_np,
            cache,
            requested_quality,
            color_threshold,
            exclude_threshold,
        )
        payload = {
            'mask': mask_array,
            'precomputed': precomputed,
            'quality': requested_quality,
        }
        if blocking or self._preview_worker is None:
            result = self._compute_preview(payload)
            self._on_preview_ready(generation, self._image_serial, result, requested_quality)
        else:
            payload.update({'generation': generation, 'image_id': self._image_serial})
            self._preview_worker.schedule(payload)

    def _on_live_preview_toggle(self):
        if not self.live_preview.get():
            self._render_left()
        elif self.preview_np is None or self.preview_dirty:
            self.preview(immediate=True)
        else:
            self._render_left()

    def _compute_preview(self, payload):
        store = payload['precomputed']
        return apply_precomputed(
            store['src'],
            store['tinted'],
            store['gate'],
            mask=payload.get('mask'),
        )

    def _on_preview_ready(self, generation, image_id, out, quality="full"):
        if image_id != self._image_serial or generation != self._preview_generation:
            return
        if quality == "draft":
            out = self._upsample_preview(out)
        self.preview_np=out
        self._last_preview_quality = quality
        self.preview_dirty = (quality != "full")
        self._draw_thumb(self.thumb_res, out)
        if self.live_preview.get():
            self._render_left()

    def _load_preferences(self):
        try:
            with open(self._prefs_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    def _save_preferences(self):
        data = {}
        for key, var in self._tracked_vars:
            try:
                value = var.get()
            except tk.TclError:
                continue
            if hasattr(value, "item"):
                value = value.item()
            data[key] = value
        data["tool"] = self._tool
        data["show_mask"] = bool(self._show_mask)
        data["geometry"] = self.geometry()
        data["range_base"] = list(self.range_base) if self.range_base is not None else None
        data["range_hsv"] = list(self.range_hsv) if self.range_hsv is not None else None
        data["exclude_base"] = list(self.exclude_base) if self.exclude_base is not None else None
        data["exclude_hsv"] = list(self.exclude_hsv) if self.exclude_hsv is not None else None
        try:
            with open(self._prefs_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except OSError:
            pass
        self._prefs = data

    def on_close(self):
        self._save_preferences()
        if self._preview_worker is not None:
            self._preview_worker.stop()
        self.destroy()

def main(): App().mainloop()
if __name__=='__main__': main()
