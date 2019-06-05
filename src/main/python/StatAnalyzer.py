import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


commentCountFile = "../../../corpus/analyzed/statCommentCount.csv"
commentDatesFile = "../../../corpus/analyzed/statCommentDates.csv"
article_fig = "../../../results/article_distribution_comments.png"
date_fig = "../../../results/article_distribution_time.png"


def main():
    # plot_all()
    plot_articles()
    plot_dates()


def plot_all():
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


def plot_articles():

    comment_count = pd.read_csv(commentCountFile)
    # sns.set_style("whitegrid")
    plt.figure(figsize=(16, 8))
    plt.bar(comment_count["noOfComments"], comment_count["articleCount"])

    plt.axis([0, 100, 0, 2000])
    # plt.text(22, 1800, 'article distribution with # of comments')
    plt.xlabel("Number of comments")
    plt.ylabel("Number of articles")
    plt.tight_layout()
    plt.savefig(article_fig, format="png")
    plt.show()


def plot_dates():
    comment_dates = pd.read_csv(commentDatesFile)
    # sns.set_style("whitegrid")
    plt.figure(figsize=(16, 8))
    x = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in comment_dates["commentDate"]]
    y = comment_dates["commentCount"]
    # plt.text(dt.datetime.strptime('2013-03-06','%Y-%m-%d').date(), 750, 'article distribution with date')
    plt.bar(x,y)
    plt.xlabel("Year")
    plt.ylabel("Number of comments")
    plt.tight_layout()
    plt.savefig(date_fig, format="png")
    plt.show()


main()
