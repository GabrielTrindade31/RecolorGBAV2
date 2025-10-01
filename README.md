# RecolorGBAV2
An app tu change the color in an image, mantain the effects on the image

# RecolorGBAV2 (Tone blend + Shading γ + Live sliders)

# RecolorGBAV2
An app to change the color in an image, maintaining the effects on the image

# RecolorGBAV2 (Tone blend + Shading γ + Live sliders)

## Why the result becomes clear
Before, we kept the luminosity/Value of the original green at the bottom. By only changing Hue/Sat, the luminosity remained high, generating a light tone. Now, there is a **tone_blend** (0..1) that recombines the luminosity with the **green colour**, preserving the **relief** (light/shadow) by a factor normalised with **shading γ**.

- `tone_blend=0` → keeps the original luminosity (previous behaviour).
- `tone_blend=1` → recombines 100% with the green colour luminosity (dark green becomes **dark** tone, maintaining the relief).
- `shading_gamma` controls the contrast of the relief (1 = neutral; >1 = more contrast).

## GUI
1) `pip install -r requirements_tk_win.txt`
2) `python run_gui.py`
3) Adjust **Tone blend** ~ 0.7–1.0 to "puddle" the green colour; adjust **Shading γ** if you want more/less contrast.
4) The preview updates **in real time** (sliders call preview).

## CLI examples
- Recolour with a given RGBA + recombination:
