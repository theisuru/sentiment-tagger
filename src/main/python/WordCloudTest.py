# coding: utf-8

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import os


newsFile = open("../../../corpus/analyzed/news.csv", "r")

# sinhalaText = "hello isuru දැන් අරක්කු ගිනි ගිනි ගණන් නිසා කසිප්පු පෙරන එක ලාභයි එක එක එක එක එක."
sinhalaText = newsFile.read()
text = sinhalaText.decode("utf-8")
cloud = WordCloud(font_path="./resources/potha.ttf",
                  width=1600,
                  height=900,
                  background_color="white",
                  # mode="RGBA",
                  regexp=r"[a-zA-Zං-𑇴']+").generate(text)

#
# cloud = WordCloud(font_path='FreeSans.otf',
#                   relative_scaling=1.0,
#                   min_font_size=4,
#                   background_color="white",
#                   width=1024,
#                   height=768,
#                   scale=3,
#                   font_step=1,
#                   collocations=False,
#                   regexp=r"[a-zA-Zං-෴']+",
#                   margin=2
#                   ).generate(sinhalaText)

plt.figure(1, figsize=(8, 8))
plt.imshow(cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

regexp = r"[a-zA-Zං-෴]+"
for t in re.findall(regexp, text, flags=re.UNICODE):
    print(t)

print(len('අරක්කු'))
print(len('එක'))
print(len('ගිනි'))
print(len('abc'))
