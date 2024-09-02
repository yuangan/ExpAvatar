<div align="center">
  
# ExpAvatar: High-Fidelity Avatar Generation of Unseen Expressions with 3D Face Priors (Official)

</div>

## Setup & Preparation

### Environment Setup

```bash
conda create -n expavatar_fpdiff python=3.8
conda activate expavatar_fpdiff
conda install pytorch=1.11 cudatoolkit=11.3 torchvision -c pytorch
conda install mpi4py dlib scikit-learn scikit-image tqdm -c conda-forge
pip install lmdb opencv-python kornia yacs blobfile chumpy face-alignment==1.3.4 pandas lpips
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
```

### Download Dataset

Download the demo dataset from [Yandex](https://disk.yandex.com/d/pxGuaG5_ehsWyQ), and organize it like:
```
----./
  |-ExpAvatar
  |-ExpAvatar.zip
```
then unzip them with:
```bash
unzip ExpAvatar.zip -d ./
cd ./ExpAvatar/stepII/
unzip baselines.zip -d ./baselines
```

### Run Inference
Run inference in `./ExpAvatar/stepII/`
```bash
bash inference.sh
```

# TODO:
- [ ] Environment setup
- [ ] Release the inference code of ExpAvatar.
- [ ] Release the metrics calculation of ExpAvatar.
- [ ] Release the processed data.
- [ ] Release the training code of ExpAvatar Step I.
- [ ] Release the training code of ExpAvatar Step II.

# Acknowledge
We acknowledge these works for their public code: [DiffusionRig](https://github.com/adobe-research/diffusion-rig), [INSTA](https://github.com/Zielon/INSTA-pytorch), [MICA's face tracker](https://github.com/Zielon/metrical-tracker), [IMavatar](https://github.com/zhengyuf/IMavatar), [PointAvatar](https://github.com/zhengyuf/PointAvatar).

