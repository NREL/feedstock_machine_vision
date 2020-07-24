# Feedstock Anomaly Detection

Developed a time series classfification model to detect anamolous feedstock.

## Model Block Diagram:
![Block Diagram](https://github.com/NREL/feedstock_machine_vision/blob/master/binary%20classification/HighLevelModelRepresentation.png)

## Run Instructions:

1. ```python imagetagger_baseline.py```

1. ```python CNNfeatureExtraction.py --model ResNet```

2. ```python preProcessData.py --windowsize 12```

3. ```python timeSeriesClassification.py```
