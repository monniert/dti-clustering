# dti-clustering

Pytorch implementation of "Deep Transformation-Invariant Clustering" 
paper:

[Preprint](https://arxiv.org/abs/2006.11132) | [Project 
webpage](http://imagine.enpc.fr/~monniert/DTIClustering)

![teaser.jpg](http://imagine.enpc.fr/~monniert/DTIClustering/teaser.jpg)

## Installation :construction_worker:

### 1. Create conda environment

```
conda env create -f environment.yml
conda activate dtic
```

**Optional:** some monitoring routines are implemented, you can use them by specifying the 
visdom port in the config file. You will need to install `visdom` from source beforehand

```
git clone https://github.com/facebookresearch/visdom
cd visdom && pip install -e .
```

### 2. Download non-torchvision datasets

```
./download_data.sh
```

## How to use :rocket:

### 1. Launch a training

```
cuda=gpu_id config=filename.yml tag=run_tag ./pipeline.sh
```

where:
- `gpu_id` is a target cuda device id,
- `filename.yml` is a YAML config located in `configs` folder,
- `run_tag` is a tag for the experiment.

Results are saved at `runs/${DATASET}/${DATE}_${run_tag}` where `DATASET` is the dataset name 
specified in `filename.yml` and `DATE` is the current date in `mmdd` format. Some training 
visual results like prototype evolutions and transformation prediction examples will be 
saved. Here is an example of learned MNIST prototypes and transformation predictions for a 
given query image:

![prototypes.gif](./demo/prototypes.gif)

![transformation.gif](./demo/transformation.gif)

### 2. Reproduce our quantitative results on MNIST-test (10 runs)

```
cuda=gpu_id config=mnist_test.yml tag=dtikmeans ./multi_pipeline.sh
```

Switch the model name to `dtigmm` in the config file to reproduce results for DTI GMM. 
Available configs are:

- affnist_test.yml
- fashion_mnist.yml
- frgc.yml
- mnist.yml
- mnist_1k.yml
- mnist_color.yml
- mnist_test.yml
- svhn.yml
- usps.yml

### 3. Reproduce our qualitative results on Instagram collections

1. Create a santaphoto dataset by running `process_insta_santa.sh` script. It can take a 
   while to scrape the 10k posts from Instagram.
2. Launch training with `cuda=gpu_id config=instagram.yml tag=santaphoto ./pipeline.sh`

That's it!

## How to cite? :clipboard:

If you find this work useful in your research, please consider citing:

```
@inproceedings{monnier2020dticlustering,
  title={{Deep Transformation-Invariant Clustering}},
  author={Monnier, Tom and Groueix, Thibault and Aubry, Mathieu},
  booktitle={NeurIPS},
  year={2020},
}
```
