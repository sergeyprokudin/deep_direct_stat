# Deep Directional Statistics: Pose Estimation with Uncertainty Quantification.


 * [Deep Directional Statistics: Pose Estimation with Uncertainty Quantification.](#deep-direct-stat)
    * [Installation](#installation)
    * [Datasets](#datasets)
        * [PASCAL3D+](#pascal3d)
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

<!--Evaluation:-->

<!--```-->
<!--matlab -nodisplay -rath('/lustre/home/sprokudin/RenderForCNN/view_estimation'); test_gt('/home/sprokudin/RenderForCNN/view_estimation/vp_test_results_mixture','/lustre/home/sprokudin/RenderForCNN/data/real_images/voc12val_easy_gt_bbox'); ; catch; end; quit;"-->
 <!---nodisplay -r "try addpath('view_estimation'); test_gt('/view_estimation/vp_test_results_mixture','data/real_images/voc12val_easy_gt_bbox'); ; catch; end; quit;"-->
<!--```-->

## Demo on PASCAL3D+

## Datasets

### PASCAL3D+

Download [the preprocessed data](https://drive.google.com/open?id=1bDcISYXmCcTqZhhCX-bhTuUCmEH1Q8YF) and place it into "$PROJECT_DIR/data" folder.

See [demo notebook](https://github.com/sergeyprokudin/deep_direct_stat/blob/master/notebooks/PASCAL3D%2B%20data%20loading%20example.ipynb)
for an example of loading.


## Training

Coming soon.


## Pre-trained Models

Coming soon.

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



