
from __future__ import annotations
import argparse, numpy as np
from .io_utils import load_rgba, save_rgba
from .colorops import recolor_preserve_shading
from .palette import extract_dominant_colors, suggest_tint_from_palette
from .transfer import transfer_palette_recolor

def parse_rgba(s: str):
    parts=[int(p.strip()) for p in s.split(',')]
    if len(parts)!=4 or any(p<0 or p>255 for p in parts):
        raise ValueError("RGBA deve ter 4 inteiros 0..255")
    return tuple(parts)

def parse_range(s: str):
    a,b=[float(x) for x in s.split(',')]; return (a,b)

def main():
    p=argparse.ArgumentParser(description="Recolor RGBA — hue range + palette transfer + tone blend")
    sub=p.add_subparsers(dest="cmd", required=True)

    pr=sub.add_parser("recolor")
    pr.add_argument("--input",required=True); pr.add_argument("--output",required=True)
    pr.add_argument("--rgba",type=parse_rgba,required=True)
    pr.add_argument("--keep",choices=["value","luminance"],default="value")
    pr.add_argument("--saturation-scale",type=float,default=1.0)
    pr.add_argument("--alpha-mode",choices=["preserve","multiply"],default="preserve")
    pr.add_argument("--hue-range",default=None,help="min,max graus (0..360)")
    pr.add_argument("--softness",type=float,default=12.0)
    pr.add_argument("--tone-blend",type=float,default=0.75)
    pr.add_argument("--shade-gamma",type=float,default=1.0)
    def run_pr(a):
        hr=parse_range(a.hue_range) if a.hue_range else None
        img=load_rgba(a.input)
        out=recolor_preserve_shading(img,a.rgba,a.keep,a.saturation_scale,a.alpha_mode,
                                     hr,a.softness,a.tone_blend,a.shade_gamma)
        save_rgba(out,a.output); print("Salvo:",a.output)
    pr.set_defaults(func=run_pr)

    pp=sub.add_parser("palette")
    pp.add_argument("--input",required=True); pp.add_argument("--k",type=int,default=4); pp.add_argument("--min-alpha",type=int,default=8)
    def run_pp(a):
        doms=extract_dominant_colors(load_rgba(a.input),k=a.k)
        for i,(rgba,ratio) in enumerate(doms,1):
            print(f"{i}. rgba{rgba}  ~ {ratio*100:.2f}%")
    pp.set_defaults(func=run_pp)

    pt=sub.add_parser("transfer")
    pt.add_argument("--input",required=True); pt.add_argument("--reference",required=True); pt.add_argument("--output",required=True)
    pt.add_argument("--k",type=int,default=4); pt.add_argument("--min-alpha",type=int,default=8)
    pt.add_argument("--keep",choices=["value","luminance"],default="value")
    pt.add_argument("--saturation-scale",type=float,default=1.0)
    pt.add_argument("--alpha-mode",choices=["preserve","multiply"],default="preserve")
    pt.add_argument("--tone-blend",type=float,default=1.0)
    pt.add_argument("--shade-gamma",type=float,default=1.0)
    def run_pt(a):
        out=transfer_palette_recolor(load_rgba(a.input), load_rgba(a.reference), a.k, a.keep, a.saturation_scale, a.alpha_mode, a.min_alpha, a.tone_blend, a.shade_gamma)
        save_rgba(out,a.output); print("Salvo:",a.output)
    pt.set_defaults(func=run_pt)

    a=p.parse_args(); a.func(a)

if __name__=="__main__":
    main()
