# Traffic_Sign_Classification_using_CNN
This project implements a  Convolutional Neural Network (CNN) to classify Traffic Signs  into 43 categories using the  Traffic Sign Recognition Benchmark  dataset.  It demonstrates the full pipeline   from preprocessing and normalization to model training, evaluation, and visualization.

## Dataset

The dataset is provided in preprocessed **pickle (.p)** files:
- `train.p` — training set  
- `valid.p` — validation set  
- `test.p` — test set  

Each file contains:
- `features`: 32×32 RGB images  
- `labels`: integer class IDs (0–42)

###  Note on Dataset Files

The dataset files are **too large to be uploaded to GitHub**.  
If you wish to run this project locally:
1. Download the original **GTSRB dataset** from [Kaggle](https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-preprocessed) or the [official GTSRB site](https://benchmark.ini.rub.de/).  
2. Preprocess them into pickle format (`train.p`, `valid.p`, `test.p`) following the same structure used here.  
3. Place them inside a folder named:traffic-signs-data/


## Model Architecture

The CNN is implemented using **Keras** (`Sequential` API) and is inspired by the classic **LeNet-5** architecture.

| Layer | Type | Parameters | Activation |
|-------|------|-------------|-------------|
| 1 | Conv2D | 6 filters, 5×5 | ReLU |
| 2 | Conv2D | 16 filters, 5×5 | ReLU |
| 3 | Flatten | — | — |
| 4 | Dense | 120 units | ReLU |
| 5 | Dense | 84 units | ReLU |
| 6 | Dense | 43 units | Softmax |

**Loss:** `sparse_categorical_crossentropy`  
**Optimizer:** `Adam(lr=0.001)`  
**Metric:** `accuracy`

---

##  Training

The model is trained on normalized grayscale images with validation accuracy tracking.

```python
history = cnn_model.fit(X_train_gray_norm,
                     y_train,
                     batch_size=128,
                     epochs=30,
                     validation_data=(X_validation_gray_norm, y_validation))
