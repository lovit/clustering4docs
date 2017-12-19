# soyclustering: Python clustering algorithm library

For document clustering, 

	from soyclustering import SphericalKMeans
	spherical_kmeans = SphericalKMeans(n_clusters=1000, max_iter=10, verbose=1,
                               init='similar_cut', sparsity='minimum_df', 
                               minimum_df_factor=0.05)

	from scipy.io import mmread
	x = mmread(mm_file).tocsr() # Only for sparse matrix

	labels = spherical_kmeans.fit_predict(x)

The kmeans algorith in scikit-learn have been implemented for Euclidean distance, but cosine similarity is required for document clusterig. We implemented Spherical kmeans clustering, a Cosine distance version of k-means. And it uses unfamiliar initialization method, 'similar_cut'. The 'kmeans++' is a most famous kmeans initialization method, and it uses $\frac{D(x_1, x_2)^2}{sum (D(x_1, x_i)^2}$ as sampling probability for initial seeds. However, in high-dimensional space, there is no-difference within large distance values. In high-dimensional space, the 'kmeans++' works as random sampling, with only exclusing nearby points. 'similar_cut' is efficient random sampling method preventing near-duplicated points. 

For interpretability, we provide clustering keyword extractor which extracted from cluster centers and labels

	from soyclustering import proportion_keywords
	centers = spherical_kmeans.cluster_centers_
	keywords = proportion_keywords(centers, labels, index2word=None) # If you have

Each center vector looks like term proportions in its documents. If a word has higher proportion than other clusters, it is a good word to describe the corresponding cluster. We choose the words that appear more frequently as keywords. 

## See more

clustering visualization (developing): https://github.com/lovit/clustering_visualization
