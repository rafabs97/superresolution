# A comparison of neural network-based super-resolution models on 3D rendered images.
Support code for training and testing the studied methods, implemented using TensorFlow 2.

**NEW (11/09/2023)**: Try our models in this Colab Notebook! [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/103bUKKtmtoPEAlzNRo1eyMpn2wCBovqq?usp=sharing)

To superresolve an image using one of the provided models, use the demo.py script:

```
python demo.py models/<model.h5> <path to image>
```

Some example images:
<br />
<br />
<p align = "center">
<img src='examples/demo.svg'> 
</p>

## Citation

If you find our work helpful for your research, please cite our publications:

```
@inproceedings{10.1007/978-3-031-44237-7_5,
author = {Berral-Soler, Rafael and Madrid-Cuevas, Francisco J. and Ventura, Sebasti\'{a}n and Mu\~{n}oz-Salinas, Rafael and Mar\'{\i}n-Jim\'{e}nez, Manuel J.},
title = {A Comparison of&nbsp;Neural Network-Based Super-Resolution Models on&nbsp;3D Rendered Images},
booktitle = {Computer Analysis of Images and Patterns: 20th International Conference, CAIP 2023, Limassol, Cyprus, September 25–28, 2023, Proceedings, Part I},
doi = {10.1007/978-3-031-44237-7_5},
year = {2023},
publisher = {Springer-Verlag},
pages = {45–55},
isbn = {978-3-031-44236-0},
}
```
