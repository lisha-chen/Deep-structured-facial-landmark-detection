# Deep-structured-facial-landmark-detection

This is the official implementation for the paper "Deep structured prediction for facial landmark detection"

## Requirement

python 3.7
tensorflow 1.15
numpy
scipy

## Pretrained models

- [Model for 300wtrain protocol](https://www.dropbox.com/sh/c47tzhdssrg9bjl/AADi0cMvhEnCPMTFPrEkuNrba?dl=0)
- [Model for 300wlptrain protocol](https://www.dropbox.com/sh/itwpw91gqxtfvw9/AABED2aIXpQy-4wxk9igxMGza?dl=0)

## Data for evaluation

[Link to data folder](https://www.dropbox.com/sh/c3r091bg1hbot5p/AADrpQLh4e0GZ4euBet2J0Vqa?dl=0)
Download this folder to replace the data folder in the repository (since some of the files are too large to be included in the repository).
Note the images used are the original images provided in the official websites listed below without any preprocessing (preprocessing such as croping and resizing is done in the evaluation code).

Note: the image paths and ground truth labels are stored in the .mat and the .tfrecords file.
You don't need .mat files to run the code. The .mat files are only a direct guidance of how to store the images.

### Links to datasets

[300W](https://ibug.doc.ic.ac.uk/resources/300-W/)
[menpo](https://ibug.doc.ic.ac.uk/resources/2nd-facial-landmark-tracking-competition-menpo-ben/)
[COFW](http://www.vision.caltech.edu/xpburgos/ICCV13/)
[300VW](https://ibug.doc.ic.ac.uk/resources/300-VW/)

Note: please follow the instructions on the official websites of the datasets for copyright and license information, etc.

## License

This code is only for research purpose.
Please follow the GPL-3.0 License if you use the code.

## Citation

>@incollection{NIPS2019_8515,
    title =     {Deep Structured Prediction for Facial Landmark Detection},
    author =    {Chen, Lisha and Su, Hui and Ji, Qiang},
    booktitle = {Advances in Neural Information Processing Systems 32},
    pages =     {2450--2460},
    year =      {2019},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.nips.cc/paper/8515-deep-structured-prediction-for-facial-landmark-detection.pdf}
  }



## Acknowledgement

The CNN backbone uses [FAN](https://github.com/1adrianb/face-alignment). It is a direct tensorflow reimplementation of the provided pytorch code.
The 3D model construction uses [non-rigid structure from motion](https://cs.stanford.edu/~ltorresa/projects/learning-nr-shape/) and [CE-CLM](https://github.com/TadasBaltrusaitis/OpenFace/tree/master/model_training/pdm_generation).
We thank the authors for providing the code. Please cite their works and ours if you use the code.


