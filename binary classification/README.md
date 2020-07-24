# Feedstock Anomaly Detection

Developed a time series classfification model to detect anamolous feedstock.

## Model Block Diagram:
![Block Diagram](https://github.nrel.gov/dsievers/FCIC/blob/master/FCIC/feedstock_machine_vision/Code_Gudavalli/data/HighLevelModelRepresentation.png)

## Run Instructions:

1. ```python imagetagger_baseline.py```

1. ```python CNNfeatureExtraction.py --model ResNet```

2. ```python preProcessData.py --windowsize 12```

3. ```python timeSeriesClassification.py```