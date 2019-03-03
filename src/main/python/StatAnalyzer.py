import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


commentCountFile = word2vec_model_name = "../../../corpus/analyzed/statCommentCount.csv"
commentDatesFile = word2vec_model_name = "../../../corpus/analyzed/statCommentDates.csv"

commentCount = pd.read_csv(commentCountFile)
commentDates = pd.read_csv(commentDatesFile)

# sns.set()
sns.set_style("whitegrid")

plt.figure(1)
plt.subplot(211)
plt.bar(commentCount["noOfComments"], commentCount["articleCount"])
plt.axis([0, 100, 0, 2000])
plt.text(22, 1800, r'article distribution with # of comments')
# plt.xlabel("number of comments")
# plt.ylabel("number of articles")


plt.subplot(212)
x = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in commentDates["commentDate"]]
y = commentDates["commentCount"]
plt.text(dt.datetime.strptime('2013-03-06','%Y-%m-%d').date(), 750, r'article distribution with date')
plt.bar(x,y)
# plt.xlabel("year")
# plt.ylabel("number of articles")

# plt.tight_layout()
plt.show()