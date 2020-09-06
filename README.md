[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/contactpose-a-dataset-of-grasps-with-object/grasp-contact-prediction-on-contactpose)](https://paperswithcode.com/sota/grasp-contact-prediction-on-contactpose?p=contactpose-a-dataset-of-grasps-with-object)


# [ContactPose](https://contactpose.cc.gatech.edu)

<figure>
<img src="readme_images/teaser.png" width="700">
<figcaption>Example ContactPose data: Contact Maps, 3D hand pose, and RGB-D grasp images for functional grasps.</figcaption>
</figure>

Code for the ContactPose dataset released in the following paper:

[ContactPose: A Dataset of Grasps with Object Contact and Hand Pose](https://contactpose.cc.gatech.edu) - 

[Samarth Brahmbhatt](https://samarth-robo.github.io/),
[Chengcheng Tang](https://scholar.google.com/citations?hl=en&user=WbG27wQAAAAJ),
[Christopher D. Twigg](https://scholar.google.com/citations?hl=en&user=aN-lQ0sAAAAJ),
[Charles C. Kemp](http://charliekemp.com/), and
[James Hays](https://www.cc.gatech.edu/~hays/),

**ECCV 2020**.

Please visit [http://contactpose.cc.gatech.edu](http://contactpose.cc.gatech.edu) to explore the dataset.

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

# [Documentation](docs/doc.md)

# Updates
- [x] Release object 3D models
- [x] Code for cropping images around hand-object
- [ ] Release contact modeling ML code
- [ ] Release data analysis and MANO fitting code
- [ ] RGB-D image background randomization support
- [ ] Release ROS code used for recording the dataset
- [x] MANO and object mesh rendering (**new**)
- [ ] Documentation using [Read the Docs](https://readthedocs.org)
