
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox
from PIL import Image, ImageTk
import numpy as np
from .selection_mask import Mask
from .colorops import recolor_rgba
from .palette import extract_palette, transfer_map
from .io_utils import pil_open, pil_to_np, np_to_pil

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Recolor RGBA — Pro v1.0")
        self.configure(bg="#0B1020"); self.geometry("1400x860")
        self._tool="Brush"; self._show_mask=True
        self.orig_np=None; self.ref_np=None; self.mask=None; self.preview_np=None
        self.zoom = 1.0
        self.apply_only=tk.BooleanVar(value=False)
        self.live_preview=tk.BooleanVar(value=True)
        self.status=tk.StringVar(value="Load an image. Left: draw/select; Right: controls. Scroll=zoom, Right-drag=pan.")
        self._rubber_id=None; self._rubber_bbox=None
        self._build(); self._bind_shortcuts()

    def _build(self):
        s=ttk.Style(self); s.theme_use("clam")
        s.configure("TFrame", background="#12172A"); s.configure("TLabel", background="#12172A", foreground="#E5E7EB")
        s.configure("TButton", background="#1F243C", foreground="#E5E7EB")
        paned = ttk.Panedwindow(self, orient="horizontal"); paned.pack(fill="both", expand=True, padx=8, pady=8)
        left = ttk.Frame(paned); right = ttk.Frame(paned); paned.add(left, weight=1); paned.add(right, weight=1)

        # LEFT
        cframe = ttk.Frame(left); cframe.pack()
        self.canvas = tk.Canvas(cframe, width=820, height=520, bg="#0E1224", highlightthickness=0, scrollregion=(0,0,820,520), cursor="tcross")
        hbar = tk.Scrollbar(cframe, orient="horizontal"); vbar = tk.Scrollbar(cframe, orient="vertical")
        hbar.config(command=self.canvas.xview); vbar.config(command=self.canvas.yview)
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0,column=0,sticky="nsew"); vbar.grid(row=0,column=1,sticky="ns"); hbar.grid(row=1,column=0,sticky="ew")
        cframe.grid_columnconfigure(0, weight=1); cframe.grid_rowconfigure(0, weight=1)

        bar=ttk.Frame(left); bar.pack(fill="x",pady=4)
        for n in ["Brush","Eraser","Rect","Ellipse","Wand","Invert","Clear","Undo","Redo","Show/Hide"]:
            ttk.Button(bar,text=n,command=lambda x=n:self._tool_action(x)).pack(side="left",padx=3)

        ctr=ttk.Frame(left); ctr.pack(fill="x",pady=6)
        self.brush_size=tk.IntVar(value=36); ttk.Label(ctr,text="Size").pack(side="left"); tk.Scale(ctr,from_=2,to=200,variable=self.brush_size,orient="horizontal",length=150,command=lambda e:self._refresh()).pack(side="left",padx=6)
        self.feather=tk.IntVar(value=12); ttk.Label(ctr,text="Feather").pack(side="left"); tk.Scale(ctr,from_=0,to=100,variable=self.feather,orient="horizontal",length=150,command=lambda e:self._refresh()).pack(side="left",padx=6)
        self.tolerance=tk.IntVar(value=24); ttk.Label(ctr,text="Tolerance").pack(side="left"); tk.Scale(ctr,from_=0,to=120,variable=self.tolerance,orient="horizontal",length=150).pack(side="left",padx=6)
        ttk.Label(ctr,text="Zoom ×").pack(side="left"); self.zoom_var = tk.DoubleVar(value=1.0)
        tk.Scale(ctr,from_=0.25,to=6.0,resolution=0.05,variable=self.zoom_var,orient="horizontal",length=220,command=self.on_zoom).pack(side="left",padx=6)

        lf=ttk.Frame(left); lf.pack(fill="x")
        ttk.Button(lf,text="Load Original…",command=self.load_original).pack(side="left",padx=3)
        ttk.Button(lf,text="Load Reference…",command=self.load_reference).pack(side="left",padx=3)
        ttk.Button(lf,text="Save PNG…",command=self.save_png).pack(side="left",padx=3)
        ttk.Checkbutton(lf, text="Apply to selection only", variable=self.apply_only, command=self.preview).pack(side="left", padx=12)
        ttk.Checkbutton(lf, text="Live preview", variable=self.live_preview, command=self.preview).pack(side="left", padx=12)

        ttk.Label(self, textvariable=self.status).pack(fill="x", side="bottom")

        # RIGHT
        cf=ttk.Labelframe(right,text="Target Color"); cf.pack(fill="x",pady=6)
        self.r=tk.IntVar(value=249); self.g=tk.IntVar(value=53); self.b=tk.IntVar(value=92); self.a=tk.IntVar(value=253)
        for (lbl,var) in [("R",self.r),("G",self.g),("B",self.b),("A",self.a)]:
            ttk.Label(cf,text=lbl).pack(side="left"); tk.Scale(cf,from_=0,to=255,variable=var,orient="horizontal",length=160,command=lambda e:self.preview()).pack(side="left",padx=6)
        ttk.Button(cf,text="Pick…",command=self.pick_color).pack(side="left",padx=6)

        tf=ttk.Labelframe(right,text="Tone / Hue"); tf.pack(fill="x",pady=6)
        self.keep=tk.StringVar(value="value"); ttk.Label(tf,text="Preserve").pack(side="left"); ttk.Combobox(tf,values=["value","luminance"],textvariable=self.keep,width=10).pack(side="left",padx=6)
        self.sat=tk.DoubleVar(value=1.0); ttk.Label(tf,text="Sat ×").pack(side="left"); tk.Scale(tf,from_=0,to=2,resolution=0.01,variable=self.sat,orient="horizontal",length=180,command=lambda e:self.preview()).pack(side="left",padx=6)
        self.alpha_mode=tk.StringVar(value="preserve"); ttk.Label(tf,text="Alpha").pack(side="left"); ttk.Combobox(tf,values=["preserve","multiply"],textvariable=self.alpha_mode,width=10).pack(side="left",padx=6)
        self.tone=tk.DoubleVar(value=0.9); ttk.Label(tf,text="Tone blend").pack(side="left"); tk.Scale(tf,from_=0,to=1,resolution=0.01,variable=self.tone,orient="horizontal",length=180,command=lambda e:self.preview()).pack(side="left",padx=6)
        self.gamma=tk.DoubleVar(value=1.0); ttk.Label(tf,text="Shading γ").pack(side="left"); tk.Scale(tf,from_=0.4,to=2,resolution=0.01,variable=self.gamma,orient="horizontal",length=180,command=lambda e:self.preview()).pack(side="left",padx=6)

        hf=ttk.Labelframe(right,text="Hue Range"); hf.pack(fill="x",pady=6)
        self.hr_enable=tk.BooleanVar(value=False)
        ttk.Checkbutton(hf,text="Enable",variable=self.hr_enable,command=self.preview).pack(side="left",padx=6)
        self.hmin=tk.IntVar(value=0); self.hmax=tk.IntVar(value=360); self.hsoft=tk.IntVar(value=12)
        ttk.Label(hf,text="Min°").pack(side="left"); tk.Scale(hf,from_=0,to=360,variable=self.hmin,orient="horizontal",length=180,command=lambda e:self.preview()).pack(side="left",padx=6)
        ttk.Label(hf,text="Max°").pack(side="left"); tk.Scale(hf,from_=0,to=360,variable=self.hmax,orient="horizontal",length=180,command=lambda e:self.preview()).pack(side="left",padx=6)
        ttk.Label(hf,text="Soft°").pack(side="left"); tk.Scale(hf,from_=0,to=45,variable=self.hsoft,orient="horizontal",length=120,command=lambda e:self.preview()).pack(side="left",padx=6)
        ttk.Button(hf, text="Reset 0–360", command=self._reset_hue_range).pack(side="left", padx=8)

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

    def _bind_shortcuts(self):
        for key,tool in [("b","Brush"),("e","Eraser"),("r","Rect"),("o","Ellipse"),("w","Wand"),("i","Invert"),("c","Clear"),("u","Undo"),("y","Redo"),("m","Show/Hide")]:
            self.bind(key, lambda e, t=tool: self._tool_action(t))
        self.bind("<Control-o>", lambda e: self.load_original())
        self.bind("<Control-Shift-O>", lambda e: self.load_reference())
        self.bind("<Control-s>", lambda e: self.save_png())
        self.bind("+", lambda e: self._zoom_step(+0.1)); self.bind("-", lambda e: self._zoom_step(-0.1)); self.bind("0", lambda e: self._reset_zoom())

    # helpers
    def _reset_hue_range(self): self.hmin.set(0); self.hmax.set(360); self.preview()
    def _zoom_step(self, dz):
        z = max(0.25, min(6.0, self.zoom + dz)); self.zoom_var.set(z); self.zoom = z; self._refresh(); self.preview()
    def _reset_zoom(self):
        if self.orig_np is None: return
        W = self.orig_np.shape[1]; z = min(1.0, 820.0/max(1,W)); self.zoom_var.set(z); self.zoom = z; self._refresh(); self.preview()
    def on_zoom(self, *_): self.zoom = float(self.zoom_var.get()); self._refresh(); self.preview()
    def on_mousewheel(self, e): delta = 1 if getattr(e,"delta",0)>0 or getattr(e,"num",0)==4 else -1; self._zoom_step(delta*0.1)
    def on_pan_start(self, e): self.canvas.scan_mark(e.x, e.y)
    def on_pan_move(self, e): self.canvas.scan_dragto(e.x, e.y, gain=1)
    def _event_to_img_xy(self, e): return int(self.canvas.canvasx(e.x) / self.zoom), int(self.canvas.canvasy(e.y) / self.zoom)

    # I/O
    def load_original(self):
        p=filedialog.askopenfilename(filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.webp")])
        if not p: return
        im=pil_open(p); npimg=pil_to_np(im); self.orig_np=npimg; H,W,_=npimg.shape
        self.mask=Mask(H,W); self.zoom = min(1.0, 820.0/max(1,W)); self.zoom_var.set(self.zoom)
        self._draw_thumb(self.thumb_orig, self.orig_np)  # "ANTES" fixo
        self.preview_np=None  # força render original até a primeira preview
        self._refresh(); self.preview()

    def load_reference(self):
        p=filedialog.askopenfilename(filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.webp")])
        if not p: return
        self.ref_np=pil_to_np(pil_open(p)); self._draw_thumb(self.thumb_ref, self.ref_np); self._clear_palette_swatches()

    def save_png(self):
        if self.preview_np is None: self.preview()
        p=filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")])
        if not p: return
        np_to_pil(self.preview_np).save(p)

    # thumbnails / render
    def _draw_thumb(self, canvas, npimg):
        if npimg is None: return
        canvas.delete("all")
        im = Image.fromarray(npimg, mode="RGBA"); W,H = im.size; w,h = int(canvas['width']), int(canvas['height'])
        scale = min(w/W, h/H, 1.0); disp = im.resize((int(W*scale), int(H*scale)), Image.NEAREST)
        tkimg = ImageTk.PhotoImage(disp); canvas.image = tkimg; canvas.create_image(0,0,image=tkimg,anchor="nw")

    def _render_left(self, overlay_mask=None):
        if self.orig_np is None: return
        # >>> use PREVIEW image if we have it (live recolor on big canvas)
        base_np = self.preview_np if (self.preview_np is not None and self.live_preview.get()) else self.orig_np
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
            self.canvas.configure(cursor="spraycan" if name in ("Brush","Eraser") else "tcross")
            self.status.set(f"Tool: {name}. Left drag. Right=pan. Scroll=zoom.")
            return
        if self.mask is None: return
        if name=="Invert": self.mask.invert()
        elif name=="Clear":
            self.mask.clear()
        elif name=="Undo": self.mask.undo()
        elif name=="Redo": self.mask.redo()
        elif name=="Show/Hide": self._show_mask=not getattr(self,"_show_mask",True)
        # After state changes, update preview and left
        self._render_live_preview()
        self.preview()

    def on_down(self,e):
        if self.orig_np is None or self.mask is None: return
        ix,iy = self._event_to_img_xy(e); self._x0,self._y0 = ix,iy
        if self._tool in ("Brush","Eraser"):
            sign=+1.0 if self._tool=="Brush" else -1.0
            self.mask.brush(ix,iy,self.brush_size.get(),self.feather.get(),sign=sign)
            self._render_live_preview()
        elif self._tool=="Wand":
            self.mask.magic_wand(self.orig_np, ix, iy, tolerance=self.tolerance.get(), sign=+1.0)
            self._render_live_preview()
        elif self._tool in ("Rect","Ellipse"):
            self._rubber_bbox=(ix,iy,ix,iy); self._rubber_id=None
            self._render_live_preview(temp_shape=True)

    def on_drag(self,e):
        if self.orig_np is None or self.mask is None: return
        ix,iy = self._event_to_img_xy(e)
        if self._tool in ("Brush","Eraser"):
            sign=+1.0 if self._tool=="Brush" else -1.0
            self.mask.brush(ix,iy,self.brush_size.get(),self.feather.get(),sign=sign)
            self._render_live_preview()
        elif self._tool in ("Rect","Ellipse") and self._rubber_bbox is not None:
            self._rubber_bbox=(self._x0,self._y0,ix,iy)
            self._render_live_preview(temp_shape=True)

    def on_up(self,e):
        if self.orig_np is None or self.mask is None: return
        ix,iy = self._event_to_img_xy(e)
        if self._tool=="Rect": self.mask.rect(self._x0,self._y0,ix,iy,self.feather.get(),sign=+1.0)
        elif self._tool=="Ellipse": self.mask.ellipse(self._x0,self._y0,ix,iy,self.feather.get(),sign=+1.0)
        if self._rubber_id is not None:
            self.canvas.delete(self._rubber_id); self._rubber_id=None; self._rubber_bbox=None
        self._render_live_preview()  # final
        self.preview()

    # live preview composer
    def _build_temp_mask(self):
        if self._tool=="Rect" and self._rubber_bbox is not None:
            x0,y0,x1,y1 = self._rubber_bbox; return self.mask._rect_alpha(x0,y0,x1,y1,self.feather.get())
        if self._tool=="Ellipse" and self._rubber_bbox is not None:
            x0,y0,x1,y1 = self._rubber_bbox; return self.mask._ellipse_alpha(x0,y0,x1,y1,self.feather.get())
        return None

    def _render_live_preview(self, temp_shape=False):
        temp = self._build_temp_mask() if temp_shape else None
        # First compute preview so left canvas can use recolored result
        if self.live_preview.get():
            self.preview(temp_mask=temp)
        # Then render left with overlay + rubberband
        self._render_left(overlay_mask=temp)

    # color + palette
    def pick_color(self):
        rgb,_ = colorchooser.askcolor()
        if rgb:
            r,g,b = map(int,rgb)
            self.r.set(r); self.g.set(g); self.b.set(b)
            self.preview()

    def _clear_palette_swatches(self):
        for child in list(self.pal_frame.pack_slaves()): child.destroy()

    def _add_swatch(self, rgba, ratio):
        sw = tk.Canvas(self.pal_frame, width=40, height=20, highlightthickness=1, highlightbackground="#fff"); sw.pack(side="left", padx=4)
        sw.create_rectangle(0,0,40,20, fill=f"#{rgba[0]:02x}{rgba[1]:02x}{rgba[2]:02x}", outline="")
        def set_color(_evt=None, col=rgba):
            self.r.set(int(col[0])); self.g.set(int(col[1])); self.b.set(int(col[2])); self.a.set(int(col[3])); self.preview()
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
        self.r.set(int(target[0])); self.g.set(int(target[1])); self.b.set(int(target[2])); self.a.set(int(target[3])); self.preview()

    # preview
    def preview(self,*_, temp_mask=None):
        if self.orig_np is None: return
        tgt=(self.r.get(), self.g.get(), self.b.get(), self.a.get())

        mask_to_use=None
        if self.apply_only.get() and self.mask is not None:
            base = self.mask.alpha if self.mask.any_selected() else None
            if temp_mask is not None:
                mask_to_use = temp_mask if base is None else np.maximum(base, temp_mask)
            else:
                # If apply_only is ON and there is NO selection, pass an empty mask so nothing changes
                if base is None:
                    H,W,_=self.orig_np.shape
                    mask_to_use = np.zeros((H,W), dtype=np.float32)
                else:
                    mask_to_use = base

        hue_range=(self.hmin.get(), self.hmax.get()) if self.hr_enable.get() else None
        out=recolor_rgba(self.orig_np, tgt, mask=mask_to_use, keep=self.keep.get(), saturation_scale=self.sat.get(),
                         alpha_mode=self.alpha_mode.get(), hue_range=hue_range, softness_deg=self.hsoft.get(),
                         tone_blend=self.tone.get(), shade_gamma=self.gamma.get())
        self.preview_np=out
        self._draw_thumb(self.thumb_res, out)
        # Also update left canvas to reflect new preview
        self._render_left()

def main(): App().mainloop()
if __name__=='__main__': main()
