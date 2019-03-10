Sentiment Analysis of Sinhala News Comments
============================================

## Summary
This project contains all submodules for building a sentiment classification workflow for Sinhala
news comments. Project contains both Java and Python source code, where,
data acquisition process is developed in Java and data analysis and classification processes are 
developed in python. Project also contains the dataset used for the analysis. 

Sentiment classification process involves following steps.
1. Crawling and collecting data
2. Initial data filtering and preparing for data labeling process
3. Labeling data (comments) on its sentiment value
4. Further pre-processing data and preparing for classification
5. Classification and result interpretation

## Details

### Downloading Word2Vec Models
Since Word2Vec models are larger than 100MB, they can't be stored in git as ordinary files. 
Therefore they are stored in git-lfs. To download them you need to install [git-lfs](https://git-lfs.github.com/).

### Directory Structure and Files
```
+-- corpus - data files and generated models  
|   +-- analyzed  
|   |   +-- saved_models  
|   +-- raw_data  
|   +-- tagged  
+-- src/main/java - Java source code (data collection, tagging and cleaning)  
+-- src/main/python - Python source code (pre-processing, analysis/classification)  
+-- src/main/resources - static resource (html, properties)  
+-- README.md    
+-- pom.xml    
```

#### Dataset
**corpus** directory contains dataset and analysed models.
raw_data: This directory contains news articles with associating comments. These files contains original news articles before any of the preprocessing steps.
analyzed: This directory contains the pre-processed file and generated models
saved_models: Holds generated models (eg. word2wec model)

### Results
```
+---------------------+----------------+----------------+----------------+----------------+
|      Algorithm      |    Accuracy    |   Precision    |     Recall     |    F1_Score    |
+---------------------+----------------+----------------+----------------+----------------+
|     Naive Bayes     | 0.776946107784 | 0.840740740741 | 0.690690690691 | 0.752902155887 |
|    Random Forest    | 0.859281437126 | 0.91159586682  | 0.794794794795 | 0.849197860963 |
| Logistic Regression | 0.867764471058 | 0.916099773243	| 0.808808808809 | 0.859117490696 |
|         SVM         | 0.869261477046 | 0.923076923077 | 0.804804804805 | 0.859893048128 |
|      RNN LSTM       | 0.864583331347 | 0.891712707182	| 0.853146853147 | 0.861719167111 |
+---------------------+----------------+----------------+----------------+----------------+
```

#### Note
Dataset used in this project is collected by crawling Sinhala online news sites, mainly www.lankadeepa.lk.


#### contact
Please contact us if you need more information.  
- Surangika Ranathunga <surangika@cse.mrt.ac.lk>
- Isuru Liyanage <theisuru@gmail.com> 