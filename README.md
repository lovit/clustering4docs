# soyclustering: Python clustering algorithm library for document clustering

문서 군집화를 위해서는 Euclidean distance 가 아닌 Cosine distance 를 이용하는 Spherical k-means 를 이용해야 합니다. 그러나 scikit-learn 의 sklearn.cluster.KMeans 는 Spherical k-means 를 제공하지 않습니다. 또한 문서 군집화 결과를 해석하기 위하여 각 클러스터의 레이블을 달아야 합니다. 

soyclustering 은 문서 군집화를 위한 spherical k-means 알고리즘과 centroid 기반 cluster labeling algorithm 을 제공합니다.

Spherical k-means 는 bag-of-words model 과 같은 sparse vector 와 Doc2Vec 과 같은 distribted repsentation 모두에서 잘 작동합니다. 하지만, cluster labeling 은 sparse vector 로 표현된 centroid vectors 를 기준으로 작동합니다.

또한 k-means 계열 군집화 알고리즘을 이용할 때 사용자는 적절한 k 를 결정해야 합니다. Silhouette score 와 같은 방법이 있지만, 이는 저차원 벡터 공간의 군집화에 적합하며, bag-of-words model 이나 distributed representation 과 같은 고차원 공간에서는 적합한 방법이 아닙니다. Uniform effect 와 같은 현상을 피할 수 있는 현실적인 방법은 예상하는 것보다 더 많은 군집의 수를 k 로 설정한 뒤, 비슷한 군집을 하나의 군집으로 후처리 과정에서 묶는 것입니다.

soyclustering 은 이를 위해 centroid vectors 의 pairwise distance matrix 를 시각화 함으로써, 현재 군집화의 결과에 중복 군집은 없는지 살펴보며, 비슷한 군집들을 하나의 군집으로 묶는 후처리 과정을 제공합니다.

그리고 k-means 의 initializer 로 이용되는 k-means++ 은 역시 저차원 벡터 공간에서 작동하는 알고리즘입니다. 고차원 공간에서는 매우 느린 initialization 성능을 보이기 때문에 soyclustering 은 이를 개선하는 fast initializer 를 제공합니다.

## Usage

토크나이징이 되어 있는 matrix market 형식의 파일을 읽습니다. Doc2Vec 과 같은 distributed representation 에 대해서도 spherical k-means 는 작동하지만, cluster labeling algorithm 은 bag-of-words model 에서만 작동합니다.

    from scipy.io import mmread
    x = mmread(mm_file).tocsr()

구현된 spherical k-means 는 아래처럼 이용할 수 있습니다. init='similar_cut' 은 고차원 벡터에서 효율적으로 작동하는 initializer 입니다. 또한 centroid 의 sparsity 를 유지하기 위해 minimum_df 방법을 이용할 수 있습니다. 그 외의 interface 는 scikit-learn 의 k-means 와 동일합니다. fit_predict 를 통하여 군집화 결과의 labels 를 얻을 수 있습니다.

    from soyclustering import SphericalKMeans
    spherical_kmeans = SphericalKMeans(
        n_clusters=1000,
        max_iter=10,
        verbose=1,
        init='similar_cut',
        sparsity='minimum_df', 
        minimum_df_factor=0.05
    )

    labels = spherical_kmeans.fit_predict(x)

군집화 결과의 해석을 위하여 cluster labeling 을 수행합니다. soyclustering 이 제공하는 proportion keywords 함수는 keyword extraction 방법에 기반하여 각 군집의 키워드를 추출합니다. input arguments 로 군집화 결과 얻는 cluster centroid vectors 와 list of str 형식으로 이뤄진 vocab list 가 필요합니다. 또한 각 군집의 크기를 측정할 수 있는 labels 를 입력해야 합니다.

    from soyclustering import proportion_keywords
    
    centers = spherical_kmeans.cluster_centers_
    idx2vocab = ['list', 'of', 'str', 'vocab']
    keywords = proportion_keywords(centers, labels, index2word=idx2vocab)

예상하는 것보다 큰 k 를 설정하면 몇 개의 군집들은 비슷한 centroid vectors 를 지닙니다. 이러한 군집이 존재하는지 확인하기 위해서는 pairwise distance matrix 를 살펴봐야 합니다.

    from soyclustering import visualize_pairwise_distance

    # visualize pairwise distance matrix
    fig = visualize_pairwise_distance(centers, max_dist=.7, sort=True)

그리고 비슷한 군집들이 있다면 이를 하나의 군집으로 묶을 수 있습니다.

    from soyclustering import merge_close_clusters

    group_centers, groups = merge_close_clusters(centers, labels, max_dist=.5)
    fig = visualize_pairwise_distance(group_centers, max_dist=.7, sort=True)

그 뒤 다시 groups 된 centroid vectors 를 살펴보면 아래의 그림과 같습니다. diagonal elements 만 진한 색이 띈다면 각각의 군집이 서로 상이하다는 의미입니다.

![](https://github.com/lovit/clustering4docs/blob/master/assets/merge_similar_clusters.png)

merge_close_clusters 함수는 centroids 가 주어지면 Cosine distance 가 최대 max_dist 를 넘지 않는 군집들을 하나의 그룹으로 묶습니다. group centroid vectors 는 원 군집의 크기 (labels) 에 비례한 weighted average of centroids 로 계산됩니다.

groups 는 각 군집이 어떤 그룹으로 묶였는지 nested list 로 표현됩니다.

    for group in groups:
        print(group)

    [0, 19, 57, 68, 88, 115, 202, 223, 229, 237]
    [1]
    [2]
    [3, 4, 5, 8, 12, 14, 16, 18, 20, 22, 26, 28, ...]
    [6, 25, 29, 32, 37, 43, 45, 48, 53, 56, 65, ...]
    [7, 17, 34, 41, 52, 59, 76, 79, 84, 87, 93, ...]
    [9, 15, 24, 47, 51, 97]
    [10, 100, 139]
    [11, 23, 251]
    ...

## See more

pyLDAvis 를 이용하면 군집화 결과를 시각적으로 해석할 수 있습니다. 이에 관련한 코드는 다음의 github 에서 제공합니다. [kmeans_to_pyLDAvis](https://github.com/lovit/kmeans_to_pyLDAvis)