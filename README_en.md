# soyclustering: Python clustering algorithm library for document clustering

This package is implementation of **Improving spherical k-means for document clustering: Fast initialization, sparse centroid projection, and efficient cluster labeling** (Kim et al., 2020).

Cosine is more effective than Euclidean when measuring the distance between two high-dimensional (or sometimes, sparse) documents vectors.
However, scikit-learn k-means package provides only Euclidean based k-means.
Additionally, labeling clusters can be very helpful for interpreting the clustering results.

Spherical k-means works well both with sparse vector representation such as Bag-of-Words model or distributed representation such as Doc2Vec or other document embedding methods.
In lower dimensional vector space, Silhouette score method is useful to define the number of clusters (`k`).
However Silhouette score method does not work well in a high-dimensional vector space such as Bag-of-Words and Doc2Vec model space.
One of the heuristic methods to define the number of clusters is to train k-means with large `k` first and subsequently merge similar ones.
This method is also useful for preventing the Uniform Effect, which causes the size of all clusters to be similar.

`soyclustering` provides Spherical k-means (k-means which uses Cosine distance as a distance metric) and keyword extraction-based clustering labeling function.
Furthermore, the function for visualizing pairwise distances between centroids will help you to check whether redundant clusters exist, allowing you to remove redundant clusters by merging them.
`soyclustering` also provides a fast initialization method that is effective in a high-dimensional vector space.
When the size of the input data is large, the initialization method sets k to be 1000.

> Add initialization comparison figure


## Usage

You can read a matrix market file as follows. Note that the file must include tokenized outputs. Although the spherical k-means function can be used for inputs in distributed representation such as Doc2Vec, our cluster labeling algorithm works only for Bag-of-Words model.

```python
from scipy.io import mmread
x = mmread(mm_file).tocsr()
```

Sperical k-means can be used as follows. init='similar_cut' indicates our initializer that is effective in a high-dimensional vector space. If you want to preserve the sparsity of the centroid vector, you can set minimum_df_factor. Other interfaces are similar to those of scikit-learn k-means function. With fit_predict, you can retrieve the labels from the clustering result.

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

When the verbose mode is set, computation speed and the level of sparsity during the initizalition and each iteration is printed.

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

Cluster labeling can be used to intrepret the clustering results. The `proportion_keywords` function of `soyclustering` uses a keyword extraction-based method to return keywords describing each cluster. For its input arguments, you need to provide cluster centroid vectors, a list of vocabularies (as str) and labels.

```python
from soyclustering import proportion_keywords

centers = spherical_kmeans.cluster_centers_
idx2vocab = ['list', 'of', 'str', 'vocab']
keywords = proportion_keywords(centers, labels, index2word=idx2vocab)
```

The table in below is the results of cluster labels from a trained k-means with k=1,000 and 1.2M documents.

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

Setting a large `k` leads to redundant clusters. You can identify these redundant clusters by carefully examining the pairwise distance between the clusters.

```python
from soyclustering import visualize_pairwise_distance

# visualize pairwise distance matrix
fig = visualize_pairwise_distance(centers, max_dist=.7, sort=True)
```

If you find redundant clusters, you can easily merge them into a single cluster.

```python
from soyclustering import merge_close_clusters

group_centers, groups = merge_close_clusters(centers, labels, max_dist=.5)
fig = visualize_pairwise_distance(group_centers, max_dist=.7, sort=True)
```

After merging, you can check the size of dark squares in the diagonal entries of the pairwise distance matrix. If the redundant clusters are indeed successfully merged, the number of dark sqaures in the diagonal entries should have been reduced.

![](https://github.com/lovit/clustering4docs/blob/master/assets/merge_similar_clusters.png)

The function `merge_close_clusters` groups similar clusters, in which the distance between them is smaller than `max_dist`.
The centroid of the new cluster is a weighted average of original centroid vectors.
From the variable `groups`, you can return the cluster indices prior and after merging.

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
