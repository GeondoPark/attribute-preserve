# Attribution Preservation in Network Compression for Reliable Network Interpretation
This repository is the official PyTorch implementation of Attribution Preservation in Network Compression for Reliable Network Interpretation by [Geondo Park](https://github.com/GeondoPark), [June Yong Yang](), [Sung Ju Hwang](http://www.sungjuhwang.com), [Eunho Yang](https://sites.google.com/site/yangeh/). To be appear in NeurIPS 2020.

## Requirements
Currently, requires following packages
- python 3.6+
- torch 1.1+
- torchvision 0.4+
- CUDA 10.0+
- SciPy (>= 0.19.1)

## Prepare Dataset
Make data folder and put the pascal data set in it

```makefolder
mkdir data
wget -nc -P ./data http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
tar -xvf ./data/VOCtrainval_11-May-2012.tar -C ./data
mv ./data/VOCtrainval_11-May-2012 ./data/VOC2012
```
## Distillation
For stochastic matching, add --stochasticity --drop_percent 0.3 arguments for all below scripts.

Overall hyperparameters for training are set by default. In folder ./pascal/distillation:
### Train teacher
```makefolder
python voc_vanilla_main.py --gpu --exp_name name --data_path ../../data/VOC2012 --imagenet_pretrained
```
### Distillation with sensitivity weighted attribution matching
```makefolder
python voc_distill_main.py --gpu --exp_name name --arch VGG16_VOC_x8 --teacher_path <saved_teacher_path> --data_path ../../data/VOC2012 --transfer_type swa --transfer-weight 100
```

## Structure Pruning (One-shot) with sensitivity weighted attribution matching
For stochastic matching, add --stochasticity --drop_percent 0.3 arguments for all below scripts.

Overall hyperparameters for training are set by default. In folder ./pascal/structure:
```makefolder
python iteratrive_prune_distill.py --gpu --exp_name name --teacher_path <saved_teacher_path> --model_path <saved-start-model-path> --data_path ../../data/VOC2012 --percent 0.3 --iter 1 --transfer_type swa --transfer-weight 100
```

## UnStructure Pruning (Iterative) with sensitivity weighted attribution matching
For stochastic matching, add --stochasticity --drop_percent 0.3 arguments for all below scripts.

Overall hyperparameters for training are set by default. In folder ./pascal/unstructure:
```makefolder
python iteratrive_prune_distill.py --gpu --exp_name name --teacher_path <saved_teacher_path> --model_path <saved-start-model-path (teacher_path)> --data_path ../../data/VOC2012 --percent 0.2 --iter 16 --transfer_type swa --transfer-weight 100
```
## Evaluation
To evaluate attribution score, grad cam, rap, lrp and ebp were implemented.

To evaluate the localization and prediction, run:

```eval
python evaluate_pred.py --gpu --data_path ../../data/VOC2012 --path <saved checkpoint> --compression <used compression method>
```
```
python evaluate_localization.py --gpu --data_path ../../data/VOC2012 --path <saved checkpoint> --compression <used compression method> --method <attribution map method> --metric auc
```
## Implementation
 - Our network pruning code is implemented based on : [rethink-network-pruning](https://github.com/Eric-mingjie/rethinking-network-pruning)
 - Implementation of attribution maps are based on : [Relative_Attributing_Propagation](https://github.com/wjNam/Relative_Attributing_Propagation)
## Citation
```
To be updated
```
