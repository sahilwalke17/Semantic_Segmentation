# Improving Semantic Segmentation via Video Prediction and Label Relaxation





Multiple GPU training and mixed precision training are supported, and the code provides examples for training and inference. For more help, type <br/>
      
    python3 train.py --help



  
## Pre-trained models
Download checkpoints to a folder `pretrained_models`. 

* [pretrained_models/cityscapes_best.pth](https://drive.google.com/file/d/1P4kPaMY-SmQ3yPJQTJ7xMGAB_Su-1zTl/view?usp=sharing)[1071MB, WideResNet38 backbone]


ImageNet Weights
* [pretrained_models/wider_resnet38.pth.tar](https://drive.google.com/file/d/1OfKQPQXbXGbWAQJj2R82x6qyz6f-1U6t/view?usp=sharing)[833MB]

Other Weights

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
 

./scripts/train_cityscapes_SEResNeXt50.sh


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
## For pixel percentage
The file Pixel_percentage.py can be used for individual pixel percents by feeding a segmented image(changing the input datapath in the file) and running the python file.



