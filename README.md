# Tomato Leaf Disease Identification System
MNSC: An Novel Deep Learning Classification Model for Tomato Leaf Disease Identification
# Project Overview 
The MNSC model proposed in this project integrates ShuffleAttention and CBAM in parallel within the MobileNetV1 architecture, and is enhanced through label smoothing regularization, thereby forming a high-performance and lightweight classifier specifically tailored for tomato leaf diseases. This model is capable of automatically identifying common types of tomato leaf diseases, offering a high-precision and lightweight solution for the classification and identification of tomato leaf diseases.
# Project Structure
```
MNSC/
├── idea.pth                   
├── _pycache_.py                 
├── model.py                   # Definition of the improved MNSC model
├── train.py                     # Model train script
├── predict.py                # Model prediction script
└── tomoto_split/
    ├── train/                 # Training set
    ├── val/                   # Validation set
    └── test/                  # Test set
```
# Core Technologies 
__1.Integration of the ShuffleAttention mechanism__：This module enhances inter-channel information interaction, enabling more effective extraction of core lesion features.
  __2.Parallel incorporation of the CBAM mechanism__:
  By operating in parallel with ShuffleAttention, CBAM complements channel-wise attention with spatial feature refinement, collectively strengthening the network’s ability to identify and highlight diagnostically relevant regions, which contributes to improved classification accuracy.  
  __3. Adoption of label-smoothed cross-entropy loss__: Replacing the conventional cross-entropy loss with its label-smoothed variant helps mitigate overfitting and enhances model generalization.
# Recommended Environment
Python 3.10.19  
PyTorch == 2.6.0 + cu124
# Dataset acquisition and structure
The dataset should be organized in the following structure:
```
dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
├── ├──class1/
│   ├── class2/
│   └── ...
├── test/
├── ├──class1/
│   ├── class2/
│   └── ...
```

Each category folder contains tomato leaf images corresponding to that category.
# Model Training 
Use the `train.py` script for model training，after changing the path:
```
python MNSC_train.py
```
# Training Parameter Description
During the training process, model weights will be automatically saved. After training, the model performance will be evaluated on the test set.
| Initial learning rate | Epoch| Batch size | 
|:------|:----:|-------:|
|0.001 | 100  | 16   |  
# Performance Evaluation
After the training is completed, the model will be evaluated on the test set, and key metrics such as accuracy will be output.
# References and contact information
The paper is in the submission stage and will update the BiBTeX citation format after its official publication. Currently, it can be temporarily cited:
```
@article{tssc_pea_disease,  
  title={MNSC: An Novel Deep Learning Classification Model for Tomato Leaf Disease Identification},  
  author={[Author's name, to be added when published]},  
  journal={[Journal name, to be supplemented after acceptance]},  
  year={2026},  
  note={Manuscript submitted for publication}  
}  
```
# Contact Information
If you encounter code running issues or academic exchange needs, please contact:  
Email: wanghuiting@huuc.edu.cn  
GitHub Issue：Submit an issue directly in this warehouse
