# Point Cloud Moment

### Installation
Required Packages:
```
tqdm
numpy
scipy
sklearn
matplotlib
itertools
open3d
plotly
pytorch
```

### Recover the discrete point cloud measures with Christoffel-Darboux Kernel

#### 2D Stanford Bunny
see [cd_aprox_cheby_2d.ipynb](cd_aprox_cheby_2d.ipynb)

<img src="./assets/monomials2d.png" width="400">
<img src="./assets/chebyshev2d.png" width="400">

#### 3D Stanford Bunny
see [cd_aprox_cheby_3d.ipynb](cd_aprox_cheby_3d.ipynb)

<img src="./assets/monomials3d.png" width="400">
<img src="./assets/chebyshev3d.png" width="400">

#### Classification on ModelNet40
see [moment_classifier.ipynb](moment_classifier.ipynb)