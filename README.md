# TAG: Tracking at Any Granularity

This is the in-progress code release for our paper, "TAG: Tracking at Any Granularity". **[[Paper](https://adamharley.com/tag/tag_draft.pdf)] [[Project Page](https://adamharley.com/tag/)]**

<img src='https://adamharley.com/tag/images/tag_zoom.gif'>

This repo will be populated with demos, evaluation scripts, dataset parsers, dataset preprocessors, and training scripts, in the weeks to come. 

Please contact Adam with anything urgent (or exciting), or open a git issue.


## Basic functionality

Provide the model with a sequence of images (3 channels) and a sequence of prompts (1 channel; usually only one of these has non-zero values), and the model will return a trajectory of coordinates, a trajectory of boxes, per-frame visibility scores, and per-frame heatmaps for multi-granularity tracking/segmentation.

```
xys_e, bboxes_e, vis_e, heatmaps_e = model(rgbs, prompts)
```


## Citation

If you use this code for your research, please cite:

Adam W. Harley, Yang You, Alex Sun, Yang Zheng, Nikhil Raghuraman, Sheldon Liang, Wen-Hsuan Chu, Suya You, Achal Dave, Pavel Tokmakov,  Rares Ambrus, Katerina Fragkiadaki, Leonidas Guibas. **TAG: Tracking at Any Granularity.** arXiv 2024.


Bibtex:
```
\inproceedings{harley2024tag
authors={Adam Harley and Yang You and Yang Zheng and Xinglong Sun and Nikhil Raghuraman and Sheldon Liang and Wen-Hsuan Chu and Suya You and Achal Dave and Pavel Tokmakov and Rares Ambrus and Katerina Fragkiadaki and Leonidas Guibas},
title={TAG: Tracking at Any Granularity},
booktitle={arXiv},
year={2024},
}
```
