# Attribution Preservation in Network Compression for Reliable Network Interpretation
This repository is the official PyTorch implementation of [Attribution Preservation in Network Compression for Reliable Network Interpretation](https://arxiv.org/abs/2010.15054) by [Geondo Park](https://github.com/GeondoPark), [June Yong Yang](), [Sung Ju Hwang](http://www.sungjuhwang.com), [Eunho Yang](https://sites.google.com/site/yangeh/), NeurIPS 2020.

## Requirements
The following requirements must be met:
- Python 3.6+
- CUDA 10.0+
- torch 1.1+
- torchvision 0.4+
- SciPy 0.19.1+
- imageio 2.8.0+

## Preparing the Dataset
Create ./data directory and extract the PASCAL VOC 2012 dataset inside.

```
mkdir data
wget -nc -P ./data http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
tar -xvf ./data/VOCtrainval_11-May-2012.tar -C ./data
```

## Training the original model(teacher)
```
python voc_vanilla_main.py --gpu --exp_name "tutorial_teacher" --data_path ../../data/VOCdevkit/VOC2012 --imagenet_pretrained
```

## Knowledge distillation
Overall hyperparameters for training are set by default. For stochastic matching, add --stochastic --drop_percent 0.3 arguments for all below scripts.

Inside directory ./pascal/distillation:

### Distillation with sensitivity weighted attribution matching
```
python voc_distill_main.py --gpu --exp_name "tutorial_distillation" --arch VGG16_VOC_x8 --teacher_path weights/tutorial_teacher/tutorial_teacher_best.pth.tar --data_path ../../data/VOCdevkit/VOC2012 --transfer_type swa --transfer_weight 100

```

## Structured Pruning (One-shot) with sensitivity weighted attribution matching
For stochastic matching, add --stochastic --drop_percent 0.3 arguments for all below scripts.

Overall hyperparameters for training are set by default. In folder ./pascal/structure:
```
python iteratrive_prune_distill.py --gpu --exp_name name --teacher_path weights/tutorial_teacher/tutorial_teacher_best.pth.tar --model_path <saved-start-model-path> --data_path ../../data/VOCdevkit/VOC2012 --percent 0.3 --iter 1 --transfer_type swa --transfer-weight 100
```

## UnStructured Pruning (Iterative) with sensitivity weighted attribution matching
For stochastic matching, add --stochastic --drop_percent 0.3 arguments for all below scripts.

Overall hyperparameters for training are set by default. In folder ./pascal/unstructure:
```
python iteratrive_prune_distill.py --gpu --exp_name name --teacher_path weights/tutorial_teacher/tutorial_teacher_best.pth.tar --model_path <saved-start-model-path (teacher_path)> --data_path ../../data/VOCdevkit/VOC2012 --percent 0.2 --iter 16 --transfer_type swa --transfer-weight 100
```

## Evaluation
To evaluate attribution scores, GradCam, LRP, EBP, and RAP were used.

To evaluate prediction and localization, run:

```
python evaluate_pred.py --gpu --data_path ../data/VOCdevkit/VOC2012 --path distillation/weights/tutorial_distill/tutorial_distill_best.pth.tar --compression kd --arch VGG16_VOC_x8
```

```
python evaluate_localization.py --gpu --data_path ../data/VOCdevkit/VOC2012 --path distillation/weights/tutorial_distill/tutorial_distill_best.pth.tar --compression kd --arch VGG16_VOC_x8 --method gcam --metric auc
```

## Implementation
 - Our network pruning code is implemented based on : [rethinking-network-pruning](https://github.com/Eric-mingjie/rethinking-network-pruning)
 - Implementation of attribution maps are based on : [Relative_Attributing_Propagation](https://github.com/wjNam/Relative_Attributing_Propagation)
 
## Citation
```
@inproceedings{park2020attribution,
  title={Attribution Preservation in Network Compression for Reliable Network Interpretation},
  author={Park, Geondo and Yang, June Yong and Hwang, Sung Ju and Yang, Eunho},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
