# GMVD Dataset [[arXiv](https://arxiv.org/abs/2109.12227)]
The GMVD dataset contributes to generalized multi-view detection with overlapping field-of-view. We build a synthetic dataset for multi-view detection using Grand theft Auto V (GTAV) and Unity Game Engine. The GMVD dataset includes seven distinct scenes, one indoor (subway) and six outdoors. One of the scenes are reserved for the test split. We vary the number of total cameras in each scene and provide different camera configurations within a scene. Additional salient features of GMVD include daytime variations (morning, afternoon, evening, night) and weather variations (sunny, cloudy, rainy, snowy). The images in the dataset are of high resolution, 1920x1080, and are synchronized. Average coverage varies from 2.76-6.4 cameras depending on the scene. The dataset can also be useful for the task of multi-view pedestrian tracking with the total of approx. 1,25,000 frames with track labels.
<img src="./extras/gmvd_dataset.png" height="300" width="1000">

|Dataset | Track Labels | IDs | # Scenes | # Training Frames | # Testing Frames | # Cameras  | # Sequences |  Avg. Coverage |
|------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------  | ------------- |  ------------- |
|WildTrack | ✓ | 313 | 1 | 360 | 40 | 7 | 1 | 3.74 |
|MultiViewX | ✓  |350 | 1 | 360 | 40 | 6 | 1 | 4.41 |
|GMVD (Ours) | ✓ | 2800 | 7 | 4983 | 1012 | 3, 5, 6, 7, 8 | 53 | 2.76 - 6.4 |

## Download
Please download the dataset from this [link](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/jeet_vora_research_iiit_ac_in/EoZySkQaB2NAuBqbyGwwwX0BP4Ma33QIWdMvlJrczeQoHQ?e=2Z7xgT).

### Folder structure
```
GMVD_Dataset
├── test
│   ├── gta_scene5
│   └── gta_scene6
└── train
    ├── gta_scene1_A
    │   ├── annotations_positions
    │   │   ├── 00000.json
    │   │   ├── 00001.json
    │   │   └── *
    │   ├── calibrations
    │   │   ├── extrinsic
    │   │   │   ├── extr_Camera1.xml
    │   │   │   ├── extr_Camera2.xml
    │   │       └── *	
    │   │   └── intrinsic
    │   │       ├── intr_Camera1.xml
    │   │       ├── intr_Camera2.xml
    │   │       └── *
    │   ├── config.json
    │   ├── gt.txt
    │   ├── Image_subsets
    │   │   ├── C1
    │   │   ├── C2
    │   │   └── *
    │   ├── matchings
    │   │   ├── Camera1_3d.txt
    │   │   ├── Camera1.txt
    │   │   ├── Camera2_3d.txt
    │   │   ├── Camera2.txt
    │   │   └── *
    │   └── rectangles.pom
    ├── gta_scene1_B
    ├── gta_scene2
    ├── gta_scene3_A
    ├── gta_scene3_B
    ├── unity_scene1
    └── unity_scene2
 ```
 
## Dataset Configurations
Each and every scene in dataset has its own configurations, specified in ```config.json``` file.
```
example :-
{ 
  "Dataset": "gta_scene1_A", 
  "num_cam": 6, 
  "num_frames": 323, 
  "img_shape": [1080, 1920], 
  "grid_shape": [800, 1200], 
  "grid_cell": 0.025, 
  "origin": [0,0], 
  "region_size":[20,30]
}

```

## Experimental Setup
* Clone this repository
* Follow this [link](https://github.com/jeetv/GMVD) to setup the environment.
* 2 Nvidia GTX 1080 Ti GPU's are been used for training with GMVD dataset.

## Training
```
[GMVD]$ python main.py -d gmvd_train -b 1 --avgpool --cls_thres 0.26
```
## Inference
* Note : --cls_thres is the parameter need to be tuned to get appropriate results.
* Download the pretrained weights from this [link](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/jeet_vora_research_iiit_ac_in/EoZySkQaB2NAuBqbyGwwwX0BP4Ma33QIWdMvlJrczeQoHQ?e=2Z7xgT)
* Inference on Wildtrack dataset
```
[GMVD]$ python main.py -d wildtrack --avgpool --resume trained_models/gmvd/Multiview_Detection_gmvd.pth --cls_thres 0.26
```
* Inference on GMVD test set
```
[GMVD]$ python main.py -d gmvd_test --avgpool --resume trained_models/gmvd/Multiview_Detection_gmvd.pth --cls_thres 0.28
```
## License and Important Note
This dataset is for educational and non-commercial research purpose only. GMVD Dataset is licensed under a [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Citation
```
@misc{vora2021bringing,
      title={Bringing Generalization to Deep Multi-view Detection}, 
      author={Jeet Vora and Swetanjal Dutta and Shyamgopal Karthik and Vineet Gandhi},
      year={2021},
      eprint={2109.12227},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
