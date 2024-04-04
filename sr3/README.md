<TOC>

# DREAM: Diffusion Rectification and Estimation-Adaptive Models (SR3 Face)

This folder is built on an offical implementation of the paper "Image Super-Resolution via Iterative Refinement([SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement))".

This repository is still under development.


## Environment configuration

The codes are based on python3.7+, CUDA version 11.0+. The specific configuration steps are as follows:

1. Create conda environment
   
   ```shell
   conda env create -f environment.yaml -n sr3 
   conda activate sr3
   ```

## Prepare dataset

Download the FFHQ (train) dataset of face via the kaggle [link] (https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq) following the command:

  ```shell
   mkdir ../datasets
   mkdir ../datasets/ffhq
   cd ../datasets/ffhq
   kaggle datasets download -d arnaud58/flickrfaceshq-dataset-ffhq
   unzip flickrfaceshq-dataset-ffhq.zip
   rm -rf flickrfaceshq-dataset-ffhq.zip
   ```

After download, preprocess the train dataset following the command:

 ```shell
   # Resize to get 16×16 LR_IMGS and 128×128 HR_IMGS, then prepare 128×128 Fake SR_IMGS by bicubic interpolation
   cd ../../sr3
   python data/prepare_data.py -p ../datasets/ffhq -o ../datasets/ffhq --size 16,128 --n_worker 8
   rm -rf ../datasets/ffhq
   ```

Preprocess the valid CelebAHQ100 dataset following the command:

 ```shell
   cd ../datasets/
   unzip celebahq_16_128.zip
   cd ../sr3
   ```

## Train
Run the following command for the training:

   ```shell
   python main.py -p train -c config/sr_sr3_16_128.json -gpu 0 --order 1
   ```

## Evaluation
Run the following command for the evaluation:

   ```shell
   python main.py -p val -c config/sr_sr3_16_128.json -gpu 0 --order 1 --resume path_to_weight_without_postfix_gen.pth
   ```

## Some pretrained results
The pretrained weights and log files can be found at [here](https://drive.google.com/drive/folders/18_lzrdwIrBJYIIi0dRPq7m8cVYjZOtHy)

## Acknowledgements
This code is mainly built on [SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement), [IDM](https://github.com/Ree1s/IDM).
