This is a deep learning test problem provided as part of the admission/selection process for UTD AI Lab research internship. (Update: I got selected!). I was provided with an image dataset (vehicle_classification.zip) that contains 8 different classes of
vehicles:
Bicycle, Bus, Car, Motorcycle, NonVehicles, Taxi, Truck, Van.
The goal is to train a CNN that can classify these different types of vehicles.
This can be divided into several steps:
1. Load the data.
2. Divide the dataset into a training set and a testing set with a ratio of 8:2.
3. Build a CNN model that takes an image (or a batch of images) as input and output the class that
the input image(s) are belonging to.
4. Train the model and keep track of the loss and accuracy.
5. Print the final accuracy.
6. (Optional bonus task) Plot the loss and accuracy curve during the training.

Approach: 
1. I first tried a simple textbook CNN model, just to set a benchmark.  On the first try I got 79.6% test accuracy on the test dataset; runtime was about 5 minutes on my computer. However, overfitting seemed to be a problem due to the consistently high difference between training loss and test loss. To reduce overfitting, I added drop_out rate and argumentation via rotation and flip. As a result, the overfitting issue got much better, but the training time took much longer due to the increase of image variants. The test accuracy improved to 81.24%; I didn't fix the random seed, so a certain amount of improvement may have been caused by randomness in the training.

<img width="753" height="786" alt="image" src="https://github.com/user-attachments/assets/48993050-af7c-46bd-9f27-73f0a92c2c4b" />


2. Then, I tried a different model structure containing more layers. The run time significantly increased to 45 minutes; test accuracy improved to 84.9%.

<img width="616" height="1405" alt="image" src="https://github.com/user-attachments/assets/4dd572f8-2913-4f4a-9b7d-620316efc654" />


3. Then, I tried another model structure which was simpler than #2 but more sophisticated than #1. The run time was about 20 minutes; test accuracy dropped to 82.58% with an observed increase in overfitting.

<img width="1015" height="1375" alt="image" src="https://github.com/user-attachments/assets/4b986705-a20b-4a38-9c30-9b0df6caaa6d" />



Based on the test results above, I chose the second model as my submission code. Here is the final output:

<img width="1405" height="941" alt="image" src="https://github.com/user-attachments/assets/bafe7af3-add9-4837-823c-04beb331ad0a" />

# Vehicle Classification (8 Classes)

This repository trains a deep learning image classifier to identify 8 vehicle-related classes:
- Bicycle
- Bus
- Car
- Motorcycle
- NonVehicles
- Taxi
- Truck
- Van

## Repository Contents
- `model.py`: main training script with static matplotlib plots after training
- `model_submit.py`: training script with live/interactive plotting during training (Jupyter-friendly)
- `model_output_PlotAfterTraining.html`: exported report with training logs and post-training plots
- `model_output_PlotDuringTraining.html`: exported report with per-epoch visual updates and final metrics

## Dataset Summary

The dataset is loaded using `torchvision.datasets.ImageFolder`
Image counts per class:

| Class | Images |
|---|---:|
| Bicycle | 1,618 |
| Bus | 2,133 |
| Car | 6,781 |
| Motorcycle | 2,986 |
| NonVehicles | 8,968 |
| Taxi | 748 |
| Truck | 2,033 |
| Van | 1,111 |
| **Total** | **26,378** |

Train/test split in scripts:

- Training: 80% (21,102 images)
- Testing: 20% (5,276 images)

## Model and Training Setup

Model included as final submission to UTD:
- `SimpleCNN2` (custom CNN with BatchNorm, ReLU, pooling, dropout)

Training settings:
- Input resolution: 64 x 64
- Batch size: 32
- Optimizer: Adam (`lr=0.001`)
- Loss: CrossEntropyLoss
- Maximum epochs: 50
- Early stopping patience: 5 (monitored by validation loss on held-out test split)

Data augmentation used:
- Resize
- Random horizontal flip
- Random rotation (15 degrees)
- Normalization

Notes:
- Pretrained EfficientNet-B0 and ResNet18 experiments are included as experimentations in the scripts.
- Best model checkpoint is saved as `best_model.pth`.

## Results

Two exported runs are included:

### Run A: Plot After Training

Source report: `model_output_PlotAfterTraining.html`

- Early stopping triggered after epoch 27
- Test Accuracy: 84.12%
- Train Accuracy: 87.58%

### Run B: Plot During Training

Source report: `model_output_PlotDuringTraining.html`

- Early stopping triggered after epoch 26
- Test Accuracy: 84.23%
- Train Accuracy: 86.83%

## Graphs and Report Artifacts
The HTML artifacts are self-contained reports that include:
- Epoch-by-epoch training logs
- Loss curves (train vs validation)
- Accuracy curves (train vs test)
- Final reported train/test accuracy


## Install dependencies
```bash
pip install torch torchvision matplotlib pandas numpy pillow ipython
```

## Reproducibility Notes
- Final metrics can vary between runs due to random initialization, stochastic augmentation, and split randomness.
