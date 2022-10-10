# SAR-AD-BagNet
  The model and code for SAR-BagNet can be found at https://github.com/xwhu002/SAR-BagNet
## Citation
```
@article{li2022sar,
  title={SAR-BagNet: An Ante-hoc Interpretable Recognition Model Based on Deep Network for SAR Image},
  author={Li, Peng and Feng, Cunqian and Hu, Xiaowei and Tang, Zixiang},
  journal={Remote Sensing},
  volume={14},
  number={9},
  pages={2150},
  year={2022},
  publisher={MDPI}
}
```

## Prerequisites
* Python (3.6+)
* Pytorch platform
* CUDA
* Windows operating system
## Dataset Preparation
* The training set should be in the folder ".../train"
* The test set should be in the folder ".../test"
* The trained model should be stored in the folder ".../saved_model"
## SAR-AD-BagNet:An Interpretable Model for SAR Image Recognition based on Adversarial Defense

## AT-based training
#### Arguments:
* ```mean```: mean of dataset
* ```std```:standard deviation of dataset
* ```epochs```:number of total epochs to run
* ```batch_size```:batch size
* ```test_batch_size```:input batch size for testing
* ```if_al```:if the adversarial learning is used in this learning,you can set False to pre-train a DNN for further training
* ```attack_strength```:attack strength of PGD attack
* ```attack_step_size```:attack step size of PGD attack
* ```attack_iter```:attack iter of PGD attack

run AT_train.py

## TRADES-based training
* ```batch_size```:batch size
* ```test_batch_size```:input batch size for testing
* ```epochs```: number of epochs to train
* ```attack_strength```:attack strength of PGD attack
* ```attack_step_size```:attack step size of PGD attack
* ```attack_iter```:attack iter of PGD attack
* ```beta```:regularization, i.e., 1/lambda in TRADES.It can be set in ```[1, 10]```. Larger ```beta``` leads to more robust and less accurate models.
* ```mean```: mean of dataset
* ```std```: standard deviation of dataset
* ```distance```: type of perturbation distance, ```'l_inf'``` or ```'l_2'```

run TRADES_train.py

## Results

Our model achieves the following performance on :

### Classification and robustness on MSTAR 10 class vehicle

| Model name                    |    Accuracy     | 
|-------------------------------|---------------- |
| 8/255 AT-based SAR-AD-BagNet  |    99.30 %      | 
| 8/255 TR-based SAR-AD-BagNet  |    98.93 %      | 
| 16/255 AT-based SAR-AD-BagNet |    99.18 %      | 
| 16/255 TR-based SAR-AD-BagNet |    98.72 %      |

 
