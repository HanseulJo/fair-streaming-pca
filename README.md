# Fair Streaming PCA

This is an official repository containing codes for our [**Fair Streaming Principal Component Analysis: Statistical and Algorithmic Viewpoint**](https://arxiv.org/abs/2310.18593), accepted at NeurIPS 2023.

## CelebA Dataset (`celeba_fair_streaming_pca/`)

### Preparation

1. Download CelebA dataset images (`img_align_celeba.zip`) from [this link](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ)
1. Download CelebA dataset annotations (`list_attr_celeba.txt`, ...) from [this link](https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs?resourcekey=0-pEjrQoTrlbjZJO2UL8K_WQ)
1. Put these to the directory `celeba_fair_streaming_pca/datasets/celeba/`. Unzip the zip file here.

### Running Codes

You may open the following four notebook files to run by yourself:

```bash
FairStreamingPCA_CelebA_RGB.ipynb
FairStreamingPCA_CelebA_grayscale.ipynb
FairStreamingPCA_CelebA_blocksizeAblation.ipynb
FairStreamingPCA_CelebA_rankAblation.ipynb
```

## UCI Dataset (`downstream_tasks_fair_streaming_pca/`)

We mostly follow the instruction in [this repository](https://github.com/amazon-science/fair-pca).

## Synthetic Experiments (`synthetic_tasks_fair_streaming_pca/`)

## Citation

If you'd like to use our code and publish a material, please cite our paper:

```bibtex
@inproceedings{
    lee2023fair,
    title={{Fair Streaming Principal Component Analysis: Statistical and Algorithmic Viewpoint}},
    author={Junghyun Lee and Hanseul Cho and Se-Young Yun and Chulhee Yun},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=TW3ipYdDQG}
}
```
