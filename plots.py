import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wordcloud
from PIL import Image


def plot_class_distribution(image_directory: str = './data/images/',
                            filename: str = 'full_data_set',
                            output_directory: str = './data/',
                            title: str = None,
                            labels=False,
                            rotate=False,
                            semilog=False):
    class_dist = {' '.join(cat_type.split()[:2]): len(os.listdir(image_directory + cat_type))
                  for cat_type in os.listdir(image_directory)}
    data_count = pd.Series(class_dist)
    # data_count = data_count[(data_count < 20000)] #  & (data_count > 500)]
    plt.figure(figsize=[5, 5])
    rect = plt.bar(data_count.index, data_count.values, color="#637b7f")
    if not labels:
        plt.gca().axes.get_xaxis().set_visible(False)
    if labels:
        if rotate:
            autolabel(rect, 90)
        else:
            autolabel(rect)
    if title:
        plt.title(title)
    plt.ylabel('Class Size')
    if semilog:
        plt.yscale('log')
        filename += '_log'
    plt.tight_layout()
    if rotate:
        plt.xticks(rotation=90)
        plt.tight_layout()
    plt.savefig(output_directory + filename + '_distribution.png', dpi=300)


def autolabel(rects, rotation=0):
    """Attach a text label above each bar in *rects*, displaying its height."""
    ax = plt.gca()
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, 0),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color=(1,1,1),
                    rotation=rotation)

def create_word_cloud(image_directory: str = './data/images/',
                            filename: str = 'full_data_set',
                            output_directory: str = './data/'):

    class_dist = {cat_type: len(os.listdir(image_directory + cat_type))
                  for cat_type in os.listdir(image_directory)}
    data_count = pd.Series(class_dist)
    data_count.index = data_count.index.map(lambda x: x.replace(' ', '\xa0'))
    #data_count = np.floor(np.log(data_count))
    # data_str = [[label]*int(value) for label, value in zip(data_count.index,
    #                                                        data_count.values)]
    mask = np.array(Image.open('cat-sil.png'))
    wc = wordcloud.WordCloud(width=mask.shape[1],
                             height=mask.shape[0],
                             background_color='white',
                             mask=mask).generate_from_frequencies(class_dist)

    plt.imshow(wc, interpolation='bilinear')
    wc.to_file(os.path.join(output_directory, filename + "_wordcloud.png"))


if __name__ == '__main__':
    plot_class_distribution(title='Major Class Imbalance', semilog=False)
    plot_class_distribution('./data/small_set/', 'small_set', title='Classes are Balanced',
                            labels=True)
    plot_class_distribution('./data/breeds/', 'breed_set', title='Classes are Reasonably Balanced',
                            labels=True, rotate=True)
    # create_word_cloud()