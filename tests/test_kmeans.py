import os
import pytest
import sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

from soyclustering import SphericalKMeans
from soyclustering import proportion_keywords
from lovit_textmining_dataset.navernews_10days import get_bow


# pytest execution with verbose
# $ pytest tests/test_kmeans.py -s -v


@pytest.fixture
def test_data():
    x, idx_to_vocab, vocab_to_idx = get_bow(date='2016-10-20', tokenize='noun')
    return {'x': x, 'idx_to_vocab': idx_to_vocab}


def test_kmeans(test_data):
    x = test_data['x']
    n_features = x.shape[1]
    n_clusters = 100
    print('\nk-means test with various configuration')

    configs = [('k-means++', None), ('similar_cut', None), ('similar_cut', 'minimum_df')]
    for config in configs:
        init, sparsity = config
        print(f'\nConfig\n  - init: {init}\n  - sparsity: {sparsity}')
        kmeans = SphericalKMeans(n_clusters=n_clusters, init=init,
            sparsity=sparsity, max_iter=5, tol=0.0001, verbose = True, random_state=0)
        labels = kmeans.fit_predict(x)
        centers = kmeans.cluster_centers_
        assert centers.shape == (n_clusters, n_features)


def test_transform(test_data):
    x = test_data['x']
    n_docs = x.shape[0]
    n_clusters = 100
    print('\ntransform test')
    kmeans = SphericalKMeans(n_clusters=n_clusters, init='similar_cut',
        sparsity=None, max_iter=5, tol=0.0001, verbose = True, random_state=0)
    kmeans.fit(x)
    distances = kmeans.transform(x)
    assert (distances.shape == (n_docs, n_clusters)) and (distances.min() >= 0) and (distances.max() <= 1)


def test_clustering_labeling(test_data):
    x = test_data['x']
    idx_to_vocab = test_data['idx_to_vocab']
    print('\nclustering labeling test')
    kmeans = SphericalKMeans(
        n_clusters=300,
        init='similar_cut',
        sparsity=None,
        max_iter=10,
        tol=0.0001,
        verbose = True,
        random_state=0
    )
    labels = kmeans.fit_predict(x)
    keywords = proportion_keywords(
        kmeans.cluster_centers_,
        labels,
        index2word=idx_to_vocab,
        topk=30,
        candidates_topk=100
    )
    for cluster_idx, keyword in enumerate(keywords):
        keyword = ' '.join([w for w,_ in keyword])
        if '아이오아이' in keyword:
            print('cluster#{} : {}'.format(cluster_idx, keyword))
