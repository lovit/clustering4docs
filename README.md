# soyclustering: Python clustering algorithm library

For document clustering, 

	from soyclustering import SphericalKMeans
	spherical_kmeans = SphericalKMeans(n_clusters=1000, max_iter=10, verbose=1)

	from scipy.io import mmread
	x = mmread(mm_file).tocsr() # Only for sparse matrix

	labels = spherical_kmeans.fit_predict(x)

## See more

clustering visualization (developing): https://github.com/lovit/clustering_visualization
