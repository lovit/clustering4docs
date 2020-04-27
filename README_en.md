# soyclustering: Python clustering algorithm library for document clustering

This package is implementation of **Improving spherical k-means for document clustering: Fast initialization, sparse centroid projection, and efficient cluster labeling** (Kim et al., 2020).

Cosine is more appropriate than Euclidean to measure the distance between two documents which are represented with high-dimensional (or sometimes, sparse) vectors.
However, scikit-learn k-means package only provides Euclidean based k-means not yet.
In addition, labeling clusters is very helpful to interpret the clustering results.

Spherical k-means works well both with sparse vector representation such as Bag-of-Words model or distributed representation such as Doc2Vec or other document embedding methods.
In lower dimensional vector space, Silhouette score method is useful to define the number of clusters (`k`).
However Silhouette score method does not works well anymore in high-dimensional vector space such as Bag-of-Words Models and Doc2Vec.
One of heuristic methods to define the cluster numbers is to train k-means with large `k` first and then merge similar ones.
The heuristic method is also useful to prevent Uniform effect which is the phenomenon that the size of all clusters leads similar.

`soyclustering` provides Spherical k-means (k-means which uses Cosine as distance metric) and keyword extraction based clustering labeling function.
And more, the function visualizing pairwise distances between centroids will help you to check whether redundant clusters exist and merging similar clusters does you to remove the redundant ones.
Also `soyclustering` provide fast initialization method which is proper to high-dimensional vector space.
The method initializes the k-means initial centroids as thousands of times when the data is large.

> Add initialization comparison figure


## Usage

토크나이징이 되어 있는 matrix market 형식의 파일을 읽습니다. Doc2Vec 과 같은 distributed representation 에 대해서도 spherical k-means 는 작동하지만, cluster labeling algorithm 은 bag-of-words model 에서만 작동합니다.

```python
from scipy.io import mmread
x = mmread(mm_file).tocsr()
```

구현된 spherical k-means 는 아래처럼 이용할 수 있습니다. init='similar_cut' 은 고차원 벡터에서 효율적으로 작동하는 initializer 입니다. 또한 centroid 의 sparsity 를 유지하기 위해 minimum_df 방법을 이용할 수 있습니다. 그 외의 interface 는 scikit-learn 의 k-means 와 동일합니다. fit_predict 를 통하여 군집화 결과의 labels 를 얻을 수 있습니다.

```python
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
```

Verbose mode 일 때에는 initialization 과 매 iteration 에서의 계산 시간과 centroid vectors 의 sparsity 가 출력됩니다.

```
initialization_time=1.218108 sec, sparsity=0.00796
n_iter=1, changed=29969, inertia=15323.440, iter_time=4.435 sec, sparsity=0.116
n_iter=2, changed=5062, inertia=11127.620, iter_time=4.466 sec, sparsity=0.108
n_iter=3, changed=2179, inertia=10675.314, iter_time=4.463 sec, sparsity=0.105
n_iter=4, changed=1040, inertia=10491.637, iter_time=4.449 sec, sparsity=0.103
n_iter=5, changed=487, inertia=10423.503, iter_time=4.437 sec, sparsity=0.103
n_iter=6, changed=297, inertia=10392.490, iter_time=4.483 sec, sparsity=0.102
n_iter=7, changed=178, inertia=10373.646, iter_time=4.442 sec, sparsity=0.102
n_iter=8, changed=119, inertia=10362.625, iter_time=4.449 sec, sparsity=0.102
n_iter=9, changed=78, inertia=10355.905, iter_time=4.438 sec, sparsity=0.102
n_iter=10, changed=80, inertia=10350.703, iter_time=4.452 sec, sparsity=0.102
```

군집화 결과의 해석을 위하여 cluster labeling 을 수행합니다. soyclustering 이 제공하는 proportion keywords 함수는 keyword extraction 방법에 기반하여 각 군집의 키워드를 추출합니다. input arguments 로 군집화 결과 얻는 cluster centroid vectors 와 list of str 형식으로 이뤄진 vocab list 가 필요합니다. 또한 각 군집의 크기를 측정할 수 있는 labels 를 입력해야 합니다.

```python
from soyclustering import proportion_keywords

centers = spherical_kmeans.cluster_centers_
idx2vocab = ['list', 'of', 'str', 'vocab']
keywords = proportion_keywords(centers, labels, index2word=idx2vocab)
```

The table in below is the results of cluster labels from trained k-means with k=1,000 and 1,226K documents.

<table>
  <colgroup>
    <col width="20%" />
    <col width="80%" />
  </colgroup>
  <thead>
    <tr class="query_and_topic">
      <th>The meaning of cluster<br>(Human label)</th>
      <th>Algorithm based extracted clustering labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td markdown="span"> The movie “Titanic" </td>
      <td markdown="span"> iceberg, zane, sinking, titanic, rose, winslet, camerons, 1997, leonardo, leo, ship, cameron, dicaprio, kate, tragedy, jack, di saster, james, romance, love, effects, special, story, people, best, ever, made </td>
    </tr>
    <tr>
      <td markdown="span"> Heros in Marvle comics (Avengers) </td>
      <td markdown="span"> zemo, chadwick, boseman, bucky, panther, holland, cap, infinity, mcu, russo, civil, bvs, antman, winter, ultron, airport, ave ngers, marvel, captain, superheroes, soldier, stark, evans, america, iron, spiderman, downey, tony, superhero, heroes </td>
    </tr>
    <tr>
      <td markdown="span"> Alien movies such as Cover-field or District 9</td>
      <td markdown="span"> skyline, jarrod, balfour, strause, invasion, independence, cloverfield, angeles, district, los, worlds, aliens, alien, la, budget, scifi, battle, cgi, day, effects, war, special, ending, bad, better, why, they, characters, their, people </td>
    </tr>
    <tr>
      <td markdown="span"> Horror movies </td>
      <td markdown="span"> gayheart, loretta, candyman, legends, urban, witt, campus, tara, reid, legend, alicia, englund, leto, rebecca, jared, scream, murders, slasher, helen, killer, student, college, students, teen, summer, cut, horror, final, sequel, scary </td>
    </tr>
    <tr>
      <td markdown="span"> The movie “The Matrix" </td>
      <td markdown="span"> neo, morpheus, neos, oracle, trinity, zion, architect, hacker, reloaded, revolutions, wachowski, fishburne, machines, agents, matrix, keanu, smith, reeves, agent, jesus, machine, computer, humans, fighting, fight, world, cool, real, special, effects </td>
    </tr>
  </tbody>
</table>

Setting `k` as large numbers leads redundant clusters. To find them, we have to examine carefully the pairwise distance between clusters.

```python
from soyclustering import visualize_pairwise_distance

# visualize pairwise distance matrix
fig = visualize_pairwise_distance(centers, max_dist=.7, sort=True)
```

If you find the redundant ones, you can easily merge them to one cluster.

```python
from soyclustering import merge_close_clusters

group_centers, groups = merge_close_clusters(centers, labels, max_dist=.5)
fig = visualize_pairwise_distance(group_centers, max_dist=.7, sort=True)
```

After grouping, you can see the size of dark squares at diagonal reduces. It means that some redundant clusters are really merged.

![](https://github.com/lovit/clustering4docs/blob/master/assets/merge_similar_clusters.png)

The function `merge_close_clusters` groups the similar clusters of which distance between them is smaller than `max_dist` together.
The centroid of new clsuter is weighted average of the centroid of merged clusters.
You can find the cluster indices before and after the merging similar clusters at variable `groups`.

```python
for group in groups:
    print(group)
```

```
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
```

## See more

In addition, the repository [`kmeans_to_pyLDAvis`](https://github.com/lovit/kmeans_to_pyLDAvis) provides k-means visualization using `PyLDAVis`

## References

- Kim, H., Kim, H. K., & Cho, S. (2020). Improving spherical k-means for document clustering: Fast initialization, sparse centroid projection, and efficient cluster labeling. Expert Systems with Applications, 150, 113288.
