import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from config_p1 import VAL_ROOT

def tsne_reduction(X):
    #print(X.shape)
    X_embedded = TSNE(n_components=2, init='random').fit_transform(X)
    #print(X_embedded.shape)
    return X_embedded

def plot_embedding(embedding, labels=None):
    labels[100] = 50
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    print(labels.min(), labels.max()+1)
    bounds = np.arange(labels.min(), labels.max()+1)
    cmap = plt.cm.jet
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    sc = ax.scatter(embedding[:,0], embedding[:,1], lw=0, s=40, c=labels, cmap=cmap, norm=norm)
    cbar = f.colorbar(sc, ax=ax, ticks=bounds)
    bounds[50] = 100
    cbar.set_ticklabels(bounds)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    ax.set_title('t-sne embedding of classification')
    plt.savefig('report_images/tsne_embedding.png')

if __name__ == '__main__':
    val_imgs = list(os.listdir(VAL_ROOT))
    val_labels = np.array([int(a.split('_')[0]) for a in val_imgs])
    #print(val_labels)
    #X = np.load('features.npy').astype(np.float64)
    #X_embedded = tsne_reduction(X)
    #np.save('tsne_embedding.npy', X_embedded)
    X_embedding = np.load('report_images/tsne_embedding.npy').astype(np.float64)
    plot_embedding(X_embedding, labels=val_labels)