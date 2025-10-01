
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox

from .colorops import recolor_preserve_shading
from .palette import extract_dominant_colors, suggest_tint_from_palette
from .transfer import transfer_palette_recolor

PRIMARY = "#6366F1"; BG="#0B1020"; CARD="#12172A"; MUTED="#9CA3AF"; TEXT="#E5E7EB"

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Recolor RGBA — Pro (Hue Range + Palette Transfer + Tone Blend)")
        self.geometry("1280x820"); self.configure(bg=BG)
        style = ttk.Style(self); style.theme_use('clam')
        for n in ("TFrame","TLabel"): style.configure(n, background=BG, foreground=TEXT)
        style.configure("Card.TFrame", background=CARD)
        style.configure("Muted.TLabel", foreground=MUTED, background=CARD)
        style.configure("Accent.TButton", background=PRIMARY, foreground="white", borderwidth=0)
        style.map("Accent.TButton", background=[("active","#7C82F5")])
        style.configure("TButton", background="#1F243C", foreground=TEXT, borderwidth=0)
        style.map("TButton", background=[("active","#242B4A")])

        body = ttk.Frame(self); body.pack(fill="both", expand=True, padx=16, pady=16)

        left = ttk.Frame(body); left.pack(side="left", fill="y")
        ttk.Label(left, text="Original", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.orig_canvas = self._img_view(left); self._gap(left)
        ttk.Button(left, text="Carregar Original...", command=self.load_original).pack(fill="x", pady=(0,8))

        ttk.Label(left, text="Referência (opcional)", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.ref_canvas = self._img_view(left); self._gap(left)
        ttk.Button(left, text="Carregar Referência...", command=self.load_reference).pack(fill="x", pady=(0,8))
        ttk.Button(left, text="Extrair Paleta (top 4)", command=self.extract_palette).pack(fill="x")

        right = ttk.Frame(body); right.pack(side="left", fill="both", expand=True, padx=(16,0))
        ttk.Label(right, text="Prévia / Resultado", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.out_canvas = self._img_view(right, big=True)

        ctrl = ttk.Frame(right, style="Card.TFrame"); ctrl.pack(fill="x", pady=(10,0))
        pad=dict(padx=8, pady=6)
        ttk.Label(ctrl, text="HEX alvo").grid(row=0,column=0, sticky="w", **pad)
        self.hex = tk.StringVar(value="#6B3FA0")
        ttk.Entry(ctrl, textvariable=self.hex, width=10).grid(row=0,column=1, **pad)
        ttk.Button(ctrl, text="🎨", width=3, command=self.pick_color).grid(row=0,column=2, **pad)

        ttk.Label(ctrl, text="RGBA").grid(row=0,column=3, sticky="e", **pad)
        self.r=tk.IntVar(value=107); self.g=tk.IntVar(value=63); self.b=tk.IntVar(value=160); self.a=tk.IntVar(value=255)
        ttk.Entry(ctrl, textvariable=self.r, width=4).grid(row=0,column=4, **pad)
        ttk.Entry(ctrl, textvariable=self.g, width=4).grid(row=0,column=5, **pad)
        ttk.Entry(ctrl, textvariable=self.b, width=4).grid(row=0,column=6, **pad)
        ttk.Entry(ctrl, textvariable=self.a, width=4).grid(row=0,column=7, **pad)

        ttk.Label(ctrl, text="Preservar").grid(row=1,column=0, sticky="w", **pad)
        self.keep=tk.StringVar(value="value")
        ttk.Combobox(ctrl, textvariable=self.keep, values=["value","luminance"], width=12, state="readonly").grid(row=1,column=1, **pad)

        ttk.Label(ctrl, text="Saturação x").grid(row=1,column=2, sticky="e", **pad)
        self.sat=tk.DoubleVar(value=1.0)
        ttk.Scale(ctrl, from_=0.0, to=2.0, orient="horizontal", variable=self.sat, command=lambda v:self.request_preview()).grid(row=1,column=3, columnspan=2, sticky="we", **pad)
        ctrl.columnconfigure(4, weight=1)

        ttk.Label(ctrl, text="Alpha").grid(row=1,column=5, sticky="e", **pad)
        self.alpha_mode=tk.StringVar(value="preserve")
        ttk.Combobox(ctrl, textvariable=self.alpha_mode, values=["preserve","multiply"], width=12, state="readonly").grid(row=1,column=6, columnspan=2, **pad)

        # Tone controls
        tone = ttk.Frame(right, style="Card.TFrame"); tone.pack(fill="x", pady=(10,0))
        ttk.Label(tone, text="Reancorar luminância na cor alvo (Tone blend)").grid(row=0,column=0, sticky="w", padx=10, pady=8)
        self.tone=tk.DoubleVar(value=0.75)
        ttk.Scale(tone, from_=0.0, to=1.0, variable=self.tone, orient="horizontal", command=lambda v:self.request_preview()).grid(row=0,column=1, sticky="we", padx=10)
        ttk.Label(tone, text="Shading γ").grid(row=0,column=2, sticky="e", padx=6)
        self.gamma=tk.DoubleVar(value=1.0)
        ttk.Scale(tone, from_=0.4, to=2.0, variable=self.gamma, orient="horizontal", command=lambda v:self.request_preview()).grid(row=0,column=3, sticky="we", padx=10)
        tone.columnconfigure(1, weight=1); tone.columnconfigure(3, weight=1)

        # Hue-range
        hr = ttk.Frame(right, style="Card.TFrame"); hr.pack(fill="x", pady=(10,0))
        ttk.Label(hr, text="Faixa de Matiz (opcional)").grid(row=0,column=0, sticky="w", padx=10, pady=8)
        self.hr_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(hr, text="Ativar", variable=self.hr_enabled, command=self.request_preview).grid(row=0,column=1, padx=6)
        ttk.Label(hr, text="Min°").grid(row=0,column=2); self.hr_min=tk.IntVar(value=80)
        ttk.Scale(hr, from_=0, to=360, variable=self.hr_min, orient="horizontal", command=lambda v:self.request_preview()).grid(row=0,column=3, sticky="we", padx=6)
        ttk.Label(hr, text="Max°").grid(row=0,column=4); self.hr_max=tk.IntVar(value=160)
        ttk.Scale(hr, from_=0, to=360, variable=self.hr_max, orient="horizontal", command=lambda v:self.request_preview()).grid(row=0,column=5, sticky="we", padx=6)
        ttk.Label(hr, text="Suavização°").grid(row=0,column=6); self.hr_soft=tk.IntVar(value=12)
        ttk.Scale(hr, from_=0, to=45, variable=self.hr_soft, orient="horizontal", command=lambda v:self.request_preview()).grid(row=0,column=7, sticky="we", padx=6)
        for c in (3,5,7): hr.columnconfigure(c, weight=1)

        actions = ttk.Frame(right, style="TFrame"); actions.pack(fill="x", pady=(8,4))
        ttk.Button(actions, text="Pré-visualizar", style="Accent.TButton", command=self.refresh_preview).pack(side="left", padx=(0,8))
        ttk.Button(actions, text="Salvar PNG...", command=self.save_png).pack(side="left")

        pal = ttk.Frame(right, style="Card.TFrame"); pal.pack(fill="x", pady=(10,0))
        ttk.Label(pal, text="Paleta extraída (clique para usar)").pack(anchor="w", padx=10, pady=(8,2))
        sw = ttk.Frame(pal, style="Card.TFrame"); sw.pack(padx=10, pady=(0,8), fill="x")
        self.swatches=[]
        for i in range(4):
            c = tk.Canvas(sw, width=140, height=42, bg=CARD, highlightthickness=0, cursor="hand2")
            c.pack(side="left", padx=6, pady=6)
            c.bind("<Button-1>", lambda e, idx=i: self.click_swatch(idx))
            self.swatches.append(c)
        self.rgba_text = tk.Text(pal, height=4, bg=CARD, fg=TEXT, relief="flat")
        self.rgba_text.pack(fill="x", padx=10, pady=(0,10))
        tbar = ttk.Frame(pal, style="Card.TFrame"); tbar.pack(fill="x", padx=10, pady=(0,10))
        ttk.Button(tbar, text="Combinar com referência (cor mais saturada)", command=self.match_reference).pack(side="left")
        ttk.Button(tbar, text="Transferência de paleta 4→4", style="Accent.TButton", command=self.apply_palette_transfer).pack(side="left", padx=8)

        self.orig_img=None; self.ref_img=None; self.out_preview=None; self.palette=[]
        self._debounce = None

    def _img_view(self, parent, big=False):
        f=ttk.Frame(parent, style="Card.TFrame"); f.pack(pady=(6,8), fill="x")
        w=560 if big else 420; h=380 if big else 240
        lab=tk.Label(f, bg=CARD); lab.pack(padx=10, pady=10); lab._wh=(w,h); return lab
    def _gap(self, p): ttk.Frame(p, height=8).pack()
    def _update_img(self, widget, pil_img):
        if pil_img is None: widget.config(image=''); return
        img=pil_img.convert("RGBA"); w,h=img.size; mw,mh=widget._wh
        s=min(mw/w, mh/h, 1.0); 
        if s<1.0: img=img.resize((int(w*s), int(h*s)), Image.LANCZOS)
        tkimg=ImageTk.PhotoImage(img); widget.image=tkimg; widget.config(image=tkimg)

    def request_preview(self):
        if self._debounce: self.after_cancel(self._debounce)
        self._debounce = self.after(120, self.refresh_preview)

    def pick_color(self):
        rgb, hx = colorchooser.askcolor(initialcolor=self.hex.get(), title="Escolher cor")
        if hx:
            self.hex.set(hx); r,g,b=[int(c) for c in rgb]; self.r.set(r); self.g.set(g); self.b.set(b); self.request_preview()

    def load_original(self):
        p = filedialog.askopenfilename(filetypes=[("Imagens","*.png;*.jpg;*.jpeg;*.webp")])
        if not p: return
        from PIL import Image
        self.orig_img = Image.open(p).convert("RGBA")
        self._update_img(self.orig_canvas, self.orig_img); self.refresh_preview()

    def load_reference(self):
        p = filedialog.askopenfilename(filetypes=[("Imagens","*.png;*.jpg;*.jpeg;*.webp")])
        if not p: return
        from PIL import Image
        self.ref_img = Image.open(p).convert("RGBA")
        self._update_img(self.ref_canvas, self.ref_img)
        self.extract_palette(auto=True)

    def refresh_preview(self):
        if self.orig_img is None: return
        r,g,b,a = self.r.get(), self.g.get(), self.b.get(), self.a.get()
        tgt=(max(0,min(255,r)), max(0,min(255,g)), max(0,min(255,b)), max(0,min(255,a)))
        arr = np.array(self.orig_img.convert("RGBA"))
        hr = (self.hr_min.get(), self.hr_max.get()) if self.hr_enabled.get() else None
        out = recolor_preserve_shading(arr, tgt, keep_channel=self.keep.get(),
                                       saturation_scale=self.sat.get(), alpha_mode=self.alpha_mode.get(),
                                       hue_range_deg=hr, softness_deg=self.hr_soft.get() if hr else 0,
                                       tone_blend=self.tone.get(), shade_gamma=self.gamma.get())
        from PIL import Image
        self.out_preview = Image.fromarray(out, mode="RGBA")
        self._update_img(self.out_canvas, self.out_preview)

    def save_png(self):
        if self.out_preview is None: self.refresh_preview()
        if self.out_preview is None: return
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")])
        if not p: return
        self.out_preview.save(p, format="PNG"); messagebox.showinfo("OK", f"Salvo em:\n{p}")

    def extract_palette(self, auto=False):
        if self.ref_img is None:
            if not auto: messagebox.showinfo("Ops", "Carregue uma imagem de referência primeiro.")
            return
        arr = np.array(self.ref_img.convert("RGBA"))
        self.palette = extract_dominant_colors(arr, k=4, min_alpha=8)
        self.rgba_text.delete("1.0","end"); lines=[]
        for i in range(4):
            if i < len(self.palette):
                rgba, ratio = self.palette[i]
                hx = f"#{rgba[0]:02X}{rgba[1]:02X}{rgba[2]:02X}"
                self._draw_swatch(self.swatches[i], hx, f"rgba{rgba}  {ratio*100:.1f}%")
                lines.append(f"{i+1}. rgba{rgba}  ~ {ratio*100:.2f}%")
            else:
                self._draw_swatch(self.swatches[i], "#1F243C", f"Slot {i+1}")
        self.rgba_text.insert("end", "\n".join(lines))

    def match_reference(self):
        if self.ref_img is None:
            messagebox.showinfo("Ops", "Carregue uma imagem de referência primeiro."); return
        if not self.palette: self.extract_palette(auto=True)
        tint = suggest_tint_from_palette(self.palette)
        if tint:
            r,g,b,a = tint
            self.r.set(r); self.g.set(g); self.b.set(b); self.a.set(a)
            self.hex.set(f"#{r:02X}{g:02X}{b:02X}")
            self.refresh_preview()

    def apply_palette_transfer(self):
        if self.orig_img is None or self.ref_img is None:
            messagebox.showinfo("Ops", "Carregue original e referência para transferir a paleta.")
            return
        orig = np.array(self.orig_img.convert("RGBA"))
        ref  = np.array(self.ref_img.convert("RGBA"))
        out = transfer_palette_recolor(orig, ref, k=4, keep_channel=self.keep.get(),
                                       saturation_scale=self.sat.get(), alpha_mode=self.alpha_mode.get(),
                                       min_alpha=8, tone_blend=self.tone.get(), shade_gamma=self.gamma.get())
        from PIL import Image
        self.out_preview = Image.fromarray(out, mode="RGBA")
        self._update_img(self.out_canvas, self.out_preview)

    def click_swatch(self, idx):
        if self.palette and idx < len(self.palette):
            r,g,b,a = self.palette[idx][0]
            self.r.set(r); self.g.set(g); self.b.set(b); self.a.set(a)
            self.hex.set(f"#{r:02X}{g:02X}{b:02X}")
            self.refresh_preview()

    def _draw_swatch(self, canvas, color, label):
        canvas.delete("all")
        canvas.create_rectangle(2,2,138,28, fill=color, outline="#0E1224")
        canvas.create_text(70,36, text=label, fill=MUTED, font=("Segoe UI",9))
