import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from collections import Counter


class Visualizer:
  def __init__(self, data):
    self.data = data

  def hist_samples(self):
    plt.title("Number of sample")
    sns.countplot(x=self.data["Labels"], data=self.data)

  def hist_length(self, col="Contents"):
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.title("Distribution of news length")
    for label in self.data["Labels"].unique():
      plt.hist(self.data[self.data['Labels'] == label]
               ["Length"], bins=25, histtype=u'step')
      plt.legend(label)
      plt.show()

  def plotWordCloud(self, col="Contents"):
    for label in self.data["Labels"].unique():
      comment_words = ''
      # iterate through the csv file
      for val in self.data[self.data["Labels"] == label][col]:
          val = str(val)

          # split the value
          tokens = val.split()
          comment_words += " ".join(tokens)+" "
      wordcloud = WordCloud(width=800, height=800,
                            background_color='white',
                            contour_color='steelblue',
                            min_font_size=10).generate(comment_words)

      # plot the WordCloud image
      plt.figure(figsize=(8, 8), facecolor=None)
      plt.imshow(wordcloud, interpolation='bilinear')
      plt.axis("off")
      plt.tight_layout(pad=0)
      plt.title("word cloud {} news".format(label))
      plt.show()

  def wordBarGraphFunction(self, column):
    for label in self.data["Labels"].unique():
      topic_words = [z.lower() for y in
                     [x.split() for x in self.data[column] if isinstance(x, str)]
                     for z in y]
      word_count_dict = dict(Counter(topic_words))
      popular_words = sorted(
          word_count_dict, key=word_count_dict.get, reverse=True)
      popular_words_nonstop = [w for w in popular_words]
      plt.figure(figsize=(10, 11))
      plt.barh(range(50), [word_count_dict[w]
               for w in reversed(popular_words_nonstop[0:50])])
      plt.yticks([x + 0.5 for x in range(50)],
                 reversed(popular_words_nonstop[0:50]))
      plt.title(label)

      plt.show()
