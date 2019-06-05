# coding: utf-8

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


allData = pd.read_csv("../../../corpus/analyzed/comments_tagged_remove.csv", ";")
trainData, testData = train_test_split(allData, test_size=0.4, random_state=0)
print(trainData.columns.values)

vectorizer = CountVectorizer(analyzer="word", tokenizer=lambda text: text.split())
# vectorizer = TfidfVectorizer(analyzer="word", tokenizer=lambda text: text.split())
vectorizedTrainComments = vectorizer.fit_transform(trainData["comment"])

logisticRegressionModel = LogisticRegression()
logisticRegressionModel = logisticRegressionModel.fit(vectorizedTrainComments, trainData["label"])

vectorizedTestComments = vectorizer.transform(testData["comment"])
predictions = logisticRegressionModel.predict(vectorizedTestComments)

labelBinarizer = preprocessing.LabelBinarizer()
labelBinarizer.fit(['NEGATIVE', 'POSITIVE'])
testLabels = labelBinarizer.transform(testData["label"])
predictLabels = labelBinarizer.transform(predictions)
print(accuracy_score(testData["label"], predictions))
print(precision_score(testLabels, predictLabels))
print(f1_score(testLabels, predictLabels))


testWords = vectorizer.get_feature_names()


modelCoeffs = logisticRegressionModel.coef_.tolist()[0]
coeffdf = pd.DataFrame({'Word' : testWords, 'Coefficient' : modelCoeffs})
coeffdf = coeffdf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
print(coeffdf.tail(10))
print(coeffdf.head(10))

coeffdf.to_csv("../../../results/LR_Coefficients.csv", ",", encoding='utf-8')



