# coding: utf-8

import matplotlib.pyplot as plt
from wordcloud import WordCloud


newsFile = open("../../../corpus/analyzed/news.csv", "r")
text = newsFile.read()

cloud = WordCloud(font_path="./resources/potha.ttf",
                  width=1600,
                  height=900,
                  background_color="white",
                  # mode="RGBA",
                  regexp=r"[a-zA-Zං-𑇴']+").generate(text)

plt.figure(1, figsize=(8, 8))
plt.imshow(cloud, interpolation='bilinear')
plt.axis('off')
plt.show()
