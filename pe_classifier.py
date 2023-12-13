import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import matplotlib
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns

from utils import plot_grid_search, plot_learning_curve, plot_loss_on_dataset

PLOT_GRAPHICS = True

matplotlib.use('TkAgg')


def plot_confusion_matrix(actual, predicted):
    if not PLOT_GRAPHICS:
        return
    actual_predicted_df = pd.DataFrame({"actual": actual, "predicted": predicted, "count": np.ones(len(actual))})
    actual_predicted_df = actual_predicted_df.pivot_table(values="count", index=["actual", "predicted"], aggfunc="sum")
    actual_predicted_df = actual_predicted_df.reset_index(level=[0, 1]).pivot(index="actual", columns="predicted",
                                                                              values="count")
    sns.heatmap(actual_predicted_df, annot=True, fmt=".4g")
    plt.show()


def output_metrics(actual, predicted, classifier_name, parameters: list = None):
    mean_accuracy = np.mean(actual == predicted)
    f1_weighted = f1_score(actual, predicted, average="weighted")

    print(f"Accuracy for {classifier_name}: mean accuracy = {mean_accuracy}, f1 = {f1_weighted}. "
          f"Parameters: {parameters}")


CLASSES_LABELS = [0, 1, 2, 3]
CLASSES_STR_SELECTED = ['Trojan:Win32', 'Backdoor:Win32', 'PUA:Win32', 'not a virus']
N_GRAM_SELECTED = [(4, 4)]

df = pd.read_csv('pe_files_random.csv', header=0)
df = df.iloc[:, 1:3].dropna()

try:
    train, test = train_test_split(df, test_size=0.2)

    train_x = train.iloc[:, 0]
    train_y: pd.Series = train.iloc[:, 1]
    test_x = test.iloc[:, 0]
    test_y: pd.Series = test.iloc[:, 1]

    for n_gram in N_GRAM_SELECTED:
        start_time = time.time()

        count_vect = CountVectorizer(ngram_range=n_gram)
        tf_idf_vectorizer = TfidfTransformer()

        X_train_counts = count_vect.fit_transform(train_x)
        X_test_counts = count_vect.transform(test_x)

        X_train_tfidf = tf_idf_vectorizer.fit_transform(X_train_counts)
        X_test_tfidf = tf_idf_vectorizer.transform(X_test_counts)

        print(train_y.value_counts())
        print(test_y.value_counts())
        print(time.time() - start_time)

        start_time = time.time()
        # sgd = SGDClassifier(loss="log_loss", verbose=1, average=True)
        # sgd.fit(X_train_tfidf, train_y)
        # predicted = sgd.predict(X_test_tfidf)
        #
        # plot_learning_curve(sgd, 'SGD learning curve', X_train_tfidf, train_y)
        plot_loss_on_dataset(X_train_tfidf, train_y, "PE classification")
        # mlp = MLPClassifier(hidden_layer_sizes=(50,))
        # mlp.fit(X_train_tfidf, train_y)
        # predicted = mlp.predict(X_test_tfidf)
        # print(time.time() - start_time)
        # output_metrics(test_y, predicted, "MLP")
        # test_y_nums = test_y.map(lambda x: CLASSES_STR_SELECTED.index(x))
        # train_y_nums = train_y.map(lambda x: CLASSES_STR_SELECTED.index(x))
except Exception as e:
    print(str(e))
