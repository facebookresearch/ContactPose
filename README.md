[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/contactpose-a-dataset-of-grasps-with-object/grasp-contact-prediction-on-contactpose)](https://paperswithcode.com/sota/grasp-contact-prediction-on-contactpose?p=contactpose-a-dataset-of-grasps-with-object)

**NOTE**: We are aware that the original Dropbox data download links are not valid anymore. Most of the data is now available in IEEE DataPort. This is being tracked in [Issue 27](https://github.com/facebookresearch/ContactPose/issues/27).

For users that do not have access to IEEE DataPort, here is a [Google Drive link](https://drive.google.com/file/d/1paUAxXgHp6wDFBFw9MI1mxGElEl2KPew/view?usp=share_link) to sample data.

Instructions (assume `CONTACTPOSE_ROOT` denotes the ContactPose repository directory):
1. Delete or rename the `CONTACTPOSE_ROOT/data`.
2. Download this ZIP file to `CONTACTPOSE_ROOT`.
3. Unzip this file. It will create `CONTACTPOSE_ROOT/ContactPose sample data`. Rename this directory to `CONTACTPOSE_ROOT/data`.

# [ContactPose](https://contactpose.cc.gatech.edu)
Download and pre-processing utilities + Python dataloader for the ContactPose dataset.
The dataset was introduced in the following **ECCV 2020** paper:
[ContactPose: A Dataset of Grasps with Object Contact and Hand Pose](https://contactpose.cc.gatech.edu) - 

[Samarth Brahmbhatt](https://samarth-robo.github.io/),
[Chengcheng Tang](https://scholar.google.com/citations?hl=en&user=WbG27wQAAAAJ),
[Christopher D. Twigg](https://scholar.google.com/citations?hl=en&user=aN-lQ0sAAAAJ),
[Charles C. Kemp](http://charliekemp.com/), and
[James Hays](https://www.cc.gatech.edu/~hays/)

<figure>
<img src="readme_images/teaser.png" width="700">
<figcaption>Example ContactPose data: Contact Maps, 3D hand pose, and RGB-D grasp images for functional grasps.</figcaption>
</figure>

## Companion Repositories/Websites:
- [Explore the dataset](https://contactpose.cc.gatech.edu/contactpose_explorer.html)
- [hand-object contact ML code](https://github.com/samarth-robo/ContactPose-ML)
- [ROS code](https://github.com/samarth-robo/contactpose_ros_utils) used for recording the dataset

## Citation
```
@InProceedings{Brahmbhatt_2020_ECCV,
author = {Brahmbhatt, Samarth and Tang, Chengcheng and Twigg, Christopher D. and Kemp, Charles C. and Hays, James},
title = {{ContactPose}: A Dataset of Grasps with Object Contact and Hand Pose},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {August},
year = {2020}
}
```

# [Documentation Link](docs/doc.md)

# [Data Changelog](docs/data_changelog.md)
We have made some data and annotation corrections. The link above mentions the correction date and the exact data that was corrected.
If you got that data before the correction date, please re-download it.

# Licensing
- Code: [MIT License](LICENSE.txt)
- 3D models: each model has its own license, see `README.txt` and `licenses.json` in the [downloads](docs/doc.md#3d-models-and-3d-printing)
- All other data: [MIT License](LICENSE.txt)

# Updates
- :black_square_button: Create a HuggingFace dataset.
- :heavy_check_mark: For users that do not have access to IEEE DataPort, here is a [Google Drive link](https://drive.google.com/file/d/1paUAxXgHp6wDFBFw9MI1mxGElEl2KPew/view?usp=share_link) to sample data.
- :heavy_check_mark: Dataset has been uploaded to IEEE DataPort at https://dx.doi.org/10.21227/fb0w-gt48 to prevent Dropbox issues.
- :heavy_check_mark: [Fix annotation errors](https://github.com/facebookresearch/ContactPose/issues/7) in data from participants 31-35.
- :black_square_button: Use [rclone](https://github.com/rclone/rclone) for Dropbox downloads
- :black_square_button: [Make depth images optional in cropping script](https://github.com/facebookresearch/ContactPose/issues/6)
- :heavy_check_mark: [Robust networking utilities](utilities/networking.py) for data download with exponential backoff in case of connection failure
- :heavy_check_mark: [Speed up dataset download by organizing images into videos](docs/doc.md#download-rgb-images-only)
- :heavy_check_mark: [Release object 3D models](docs/doc.md#3d-models-and-3d-printing)
- :heavy_check_mark: [Code for cropping images around hand-object](demo.ipynb)
- :heavy_check_mark: [Release contact modeling ML code](https://github.com/samarth-robo/ContactPose-ML)
- :black_square_button: Release more data analysis code
- :heavy_check_mark: [Release MANO fitting code](utilities/mano_fitting.py) | [demo at end of notebook](demo.ipynb)
- :heavy_check_mark: [RGB-D image background randomization support](docs/doc.md#image-preprocessing)
- :heavy_check_mark: **new** Release [ROS code](https://github.com/samarth-robo/contactpose_ros_utils) used for recording the dataset
- :heavy_check_mark: [MANO and object mesh rendering](docs/rendering.md)
- :black_square_button: Documentation using [Read the Docs](https://readthedocs.org)
