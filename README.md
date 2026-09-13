# Volumetric Data Structures for Ray Marching Clouds on the GPU

This repository contains the code and resources for the bachelor's thesis *"Volumetric Data Structures for Ray Marching Clouds on the GPU: Implementation and Comparative Analysis of a Dense Grid, a Two-Level Sparse Grid, and NanoVDB."*

## Repository Structure

```
rdv/src/grid_sandbox/
├── shaders/    GLSL compute shaders for all four volumetric representations
└── scripts/    Python scripts for measurements and RDV map classes
```

## Data

The `data/` folder is not included in this repository due to file size. The original `.pt` and `.nvdb` cloud volumes used for all measurements in the thesis are available here:

[**Download cloud data (LRZ Sync+Share)**](https://syncandshare.lrz.de/getlink/fiWEeHFbALiBSC1x4B89sP/)