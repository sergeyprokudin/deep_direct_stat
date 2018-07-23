# Deep Directional Statistics: Pose Estimation with Uncertainty Quantification.


 * [Deep Directional Statistics: Pose Estimation with Uncertainty Quantification.](#deep-direct-stat)
    * [Installation](#installation)
    * [Demo on PASCAL3D+](#demo)
    * [Datasets](#datasets)
        * [PASCAL3D+](#pascal3d)
        * [IDIAP](#idiap)
        * [CAVIAR](#caviar)
        * [TownCentre](#towncentre)
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
"$PROJECT_DIR/py_env" folder) and install all necessary dependencies 
(TensorFlow, Keras, etc.).

To work with available notebooks, run:

```
bash scripts/start_notebook.sh
```

## Demo on PASCAL3D+

## Datasets

### PASCAL3D+

Download [the preprocessed data](https://drive.google.com/open?id=1bDcISYXmCcTqZhhCX-bhTuUCmEH1Q8YF) and place it into "$PROJECT_DIR/data" folder.

See [demo notebook](https://github.com/sergeyprokudin/deep_direct_stat/blob/master/notebooks/PASCAL3D%2B%20data%20loading%20example.ipynb)
for an example of loading and evaluation.

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
@conference{deepdirectstat2018,
  title = {Deep Directional Statistics: Pose Estimation with Uncertainty Quantification},
  author = {Prokudin, Sergey and Gehler, Peter and Nowozin, Sebastian},
  booktitle = {European Conference on Computer Vision (ECCV)},
  month = sep,
  year = {2018},
  month_numeric = {9}
}
```

ArXiv preprint:

 - https://arxiv.org/pdf/1805.03430.pdf

## References 

 - https://github.com/lucasb-eyer/BiternionNet (original BiternionNet repository)
 - https://github.com/ShapeNet/RenderForCNN (used for getting PASCAL3D+ dataset and evaluation)



