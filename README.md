# Deep Directional Statistics: Pose Estimation with Uncertainty Quantification.


 * [Deep Directional Statistics: Pose Estimation with Uncertainty Quantification.](#deep-direct-stat)
    * [Installation](#installation)
    * [Demo on PASCAL3D+](#demo)
    * [Datasets](#datasets)
        * [IDIAP](#idiap)
        * [CAVIAR](#caviar)
        * [PASCAL3D+](#pascal3d)
    * [Models](#models)
        * [Non-probabilistic Baseline](#non-prob)
        * [Single von Mises](#single_vm)
        * [Finite Mixture](#finite_mix)
        * [Infinite Mixture](#infinite_mix)
    * [Pre-trained Models](#pretrained)
    * [Citing](#citing)
    * [References](#refs)

## Installation  

```
bash scripts/install.sh
```

This will create a virtual environment for the project (located in 
$PROJECT_DIR/py_env folder) and install all necessary dependencies 
(TensorFlow, Keras, etc.).

## Demo on PASCAL3D+

## Datasets

### PASCAL3D+
### IDIAP
### CAVIAR
### TownCentre

## Models

### Non-probabilistic Baseline
### Single von Mises
### Finite Mixture
### Infinite Mixture

## Pre-trained Models

## Citing

```
@article{prokudin2018deep,
  title={Deep Directional Statistics: Pose Estimation with Uncertainty Quantification},
  author={Prokudin, Sergey and Gehler, Peter and Nowozin, Sebastian},
  booktitle={ECCV},
  year={2018}
}
```

ArXiv preprint:

 - https://arxiv.org/pdf/1805.03430.pdf

## References 

 - https://github.com/lucasb-eyer/BiternionNet (original BiternionNet repository)
 - https://github.com/ShapeNet/RenderForCNN (used for getting PASCAL3D+ dataset and evaluation)



