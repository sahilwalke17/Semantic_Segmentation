# Improving Semantic Segmentation via Video Prediction and Label Relaxation





Multiple GPU training and mixed precision training are supported, and the code provides examples for training and inference. For more help, type <br/>
      
    python3 train.py --help



  
## Pre-trained models
Download checkpoints to a folder `pretrained_models`. 

* [pretrained_models/cityscapes_best.pth](https://drive.google.com/file/d/1P4kPaMY-SmQ3yPJQTJ7xMGAB_Su-1zTl/view?usp=sharing)[1071MB, WideResNet38 backbone]


ImageNet Weights
* [pretrained_models/wider_resnet38.pth.tar](https://drive.google.com/file/d/1OfKQPQXbXGbWAQJj2R82x6qyz6f-1U6t/view?usp=sharing)[833MB]

Other Weights
* [pretrained_models/cityscapes_cv0_seresnext50_nosdcaug.pth](https://drive.google.com/file/d/1aGdA1WAKKkU2y-87wSOE1prwrIzs_L-h/view?usp=sharing)[324MB]
* [pretrained_models/cityscapes_cv0_wideresnet38_nosdcaug.pth](https://drive.google.com/file/d/1CKB7gpcPLgDLA7LuFJc46rYcNzF3aWzH/view?usp=sharing)[1.1GB]

## Data Loaders

Dataloaders for Cityscapes, Mapillary, Camvid and Kitti are available in [datasets](./datasets). Details of preparing each dataset can be found at [PREPARE_DATASETS.md](https://github.com/NVIDIA/semantic-segmentation/blob/master/PREPARE_DATASETS.md) <br />


## Semantic segmentation demo for a single image

If you want to try our trained model on any driving scene images, simply use

```
CUDA_VISIBLE_DEVICES=0 python demo.py --demo-image YOUR_IMG --snapshot ./pretrained_models/cityscapes_best.pth --save-dir YOUR_SAVE_DIR
```
This snapshot is trained on Cityscapes dataset, with `DeepLabV3+` architecture and `WideResNet38` backbone. The predicted segmentation masks will be saved to `YOUR_SAVE_DIR`. Check it out. 

## Semantic segmentation demo for a folder of images

If you want to try our trained model on a folder of driving scene images, simply use

```
CUDA_VISIBLE_DEVICES=0 python demo_folder.py --demo-folder YOUR_FOLDER --snapshot ./pretrained_models/cityscapes_best.pth --save-dir YOUR_SAVE_DIR
```
This snapshot is trained on Cityscapes dataset, with `DeepLabV3+` architecture and `WideResNet38` backbone. The predicted segmentation masks will be saved to `YOUR_SAVE_DIR`. Check it out. 
 
## A quick start with light SEResNeXt50 backbone

Note that, in this section, we use the standard train/val split in Cityscapes to train our model, which is `cv 0`. 

If you have less than 8 GPUs in your machine, please change `--nproc_per_node=8` to the number of GPUs you have in all the .sh files under folder `scripts`.


### Fine-tuning on Cityscapes 
Once you have the Mapillary pre-trained model (training mIoU should be 50+), you can start fine-tuning the model on Cityscapes dataset. Set `__C.DATASET.CITYSCAPES_DIR` in `config.py` to where you store the Cityscapes data. Your training mIoU in the end should be 80+. 
```
./scripts/train_cityscapes_SEResNeXt50.sh
```

### Inference

Our inference code supports two ways of evaluation: pooling and sliding based eval. The pooling based eval is faster than sliding based eval but provides slightly lower numbers. We use `sliding` as default. 
 ```
 ./scripts/eval_cityscapes_SEResNeXt50.sh <weight_file_location> <result_save_location>
 ```

In the `result_save_location` you set, you will find several folders: `rgb`, `pred`, `compose` and `diff`. `rgb` contains the color-encode predicted segmentation masks. `pred` contains what you need to submit to the evaluation server. `compose` contains the overlapped images of original video frame and the color-encode predicted segmentation masks. `diff` contains the difference between our prediction and the ground truth. 

Right now, our inference code only supports Cityscapes dataset.  


## Reproducing our results with heavy WideResNet38 backbone

Note that, in this section, we use an alternative train/val split in Cityscapes to train our model, which is `cv 2`. You can find the difference between `cv 0` and `cv 2` in the supplementary material section in our arXiv paper. 


### Fine-tuning on Cityscapes 
```
./scripts/train_cityscapes_WideResNet38.sh
```

### Inference
```
./scripts/eval_cityscapes_WideResNet38.sh <weight_file_location> <result_save_location>
```

For submitting to Cityscapes benchmark, we change it to multi-scale setting. 
 ```
 ./scripts/submit_cityscapes_WideResNet38.sh <weight_file_location> <result_save_location>
 ```

Now you can zip the `pred` folder and upload to Cityscapes leaderboard. For the test submission, there is nothing in the `diff` folder because we don't have ground truth. 

At this point, you can already achieve top performance on Cityscapes benchmark (83+ mIoU). In order to further boost the segmentation performance, we can use the augmented dataset to help model's generalization capibility. 


<table class="tg">
  <tr>
    <th class="tg-t2cw">Model Name</th>
    <th class="tg-t2cw">Mean IOU</th>
    <th class="tg-t2cw">Training Time</th>
  </tr>
  <tr>
    <td class="tg-rg0h">DeepWV3Plus(no sdc-aug)</td>
    <td class="tg-rg0h">81.4</td>
    <td class="tg-rg0h">~14 hrs</td>
  </tr>
  <tr>
    <td class="tg-rg0h">DeepSRNX50V3PlusD_m1(no sdc-aug)</td>
    <td class="tg-rg0h">80.0</td>
    <td class="tg-rg0h">~9 hrs</td>
  </tr>
</table>



