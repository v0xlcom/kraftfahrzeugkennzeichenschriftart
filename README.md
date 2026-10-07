# kraftfahrzeugkennzeichenschirftart

A German license-plate style font (FE-Schrift Art) for Ubuntu Linux. Includes `A–Z`, `Ä Ö Ü` (plus `ä ö ü`), `0–9` and space.

## Install

```bash
cp kraftfahrzeugkennzeichenschirftart.ttf ~/.local/share/fonts/
fc-cache -f ~/.local/share/fonts
```

Then select the family `kraftfahrzeugkennzeichenschirftart` in any app.

Note: TrueType stores flat outlines — the embossed-plate (3D stamp) look is applied at render time, e.g. by drawing the text twice with a slight offset (light copy underneath, dark on top).
