import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def proxy_a_distance(source_X, target_X, verbose=False):
    """
    # Compute A-distance using numpy and sklearn
    # Reference: Analysis of representations in domain adaptation, NIPS-07.
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    C_list = np.logspace(-5, 4, 10)

    half_source, half_target = int(nb_source/2), int(nb_target/2)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    best_risk = 1.0
    for C in C_list:
        clf = svm.SVC(C=C, kernel='linear', verbose=False)
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    return 2 * (1. - 2 * best_risk)


def plot_tsne_source_target(source_data, source_label, target_data, target_label, ax, name='tsne'):
    # source_data, target_data:  (None, feature_nums) numpy array
    # source_label, target_label: (None, ) numpy array
    source_size = source_data.shape[0]
    target_size = target_data.shape[0]
    features = np.concatenate([source_data, target_data], axis=0)
    labels = np.concatenate([source_label, target_label], axis=0)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    features = tsne.fit_transform(features)
    x_min, x_max = np.min(features, 0), np.max(features, 0)
    data = (features - x_min) / (x_max - x_min)
    del features
    for i in range(source_size):
        ax.text(data[i, 0], data[i, 1], 's' + str(labels[i]),
                 color=plt.cm.Set1((labels[i] + 1) / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    for i in range(source_size, source_size + target_size):
        ax.text(data[i, 0], data[i, 1], 't' + str(labels[i]),
                 color=plt.cm.Set2((labels[i] + 1) / 10.),
                 fontdict={'size': 6})
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(name)
    return ax


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    source_x, source_y = make_blobs(300, centers=[[0, 0], [0, 1]], cluster_std=0.2, random_state=0)
    target_x, target_y = make_blobs(300, centers=[[1, -1], [1, 0]], cluster_std=0.2, random_state=0)
    print('features shape ', source_x.shape)
    print('labels shape ', source_y.shape)
    fig, ax = plt.subplots()
    _ = plot_tsne_source_target(source_x, source_y, target_x, target_y, ax)
    plt.show()
