# Shape your Space: A Gaussian Mixture Regularization Approach to Deterministic Autoencoders PyTorch 

PyTorch implementation of the NeurIPS 2021 paper "Shape your Space: A Gaussian Mixture Regularization Approach to Deterministic Autoencoders". The paper can be found 
[here](https://proceedings.neurips.cc/paper/2021/hash/3c057cb2b41f22c0e740974d7a428918-Abstract.html). The code allows the users to
reproduce and extend the results reported in the paper. Please cite the
above paper when reporting, reproducing or extending the results.

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication. It will neither be
maintained nor monitored in any way.

## Setup.

1. Create a conda virtual environment
2. Clone the repository
3. Activate the environment and run 
 ```bash
cd GMM_DAE
pip install -r requirements.txt
```
## Dataset

The provided implementation is tested on MNIST, FASHION MNIST, SVHN and CELEBA images. 

For dataset preparation use the script *generate_dataset.py* in the *scripts* folder:
```bash
python scripts/generate_dataset.py <DATASETNAME> <DATASETPATH>
```
CelebA has a daily quota that can only be overcome by manually downloading the dataset and placing the following files in *\<DATASETPATH\>/celeba*:
 - *img_align_celeba.zip*
 - *list_attr_celeba.txt*
 - *identity_CelebA.txt*
 - *list_bbox_celeba.txt*
 - *list_landmarks_align_celeba.txt*
 - *list_eval_partition.txt*

The paths to the datasets need to be saved in the config.ini file.
  
## Usage

To run the code clone the repository and then run

```bash
python train.py <DATASETNAME> eg: MNIST, FASHIONMNIST, SVHN or CELEB
```
For FID computation we used the github repo [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
## License

GMM_DAE is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

