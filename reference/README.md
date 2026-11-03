# Reference data

`sa_reference_alpha10.csv` - Spalart-Allmaras CL / CD at alpha = 10 deg, Re = 6e6, M = 0.15,
transcribed from the NASA Turbulence Modeling Resource page
<https://tmbwg.github.io/turbmodels/naca0012_val_sa.html> (first six codes: same 897x257 grid;
GGNS: grid-adaptive). The spread between codes (about +-0.5 % in CL, +-1 % in CD) is the realistic
target band for any SA solver; compare against **CFL3D** and report the band.

