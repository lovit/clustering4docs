from description import __version__, __author__
from setuptools import setup, find_packages

setup(
   name="soyclustering",
   version=__version__,
   author=__author__,
   author_email='soy.lovit@gmail.com',
   url='https://github.com/lovit/clustering4docs',
   description="Python library for document clustering",
   long_description="""Python library for document clustering

It includes
- Spherical k-means and fast initializer for sparse vector representation such as bag of words.
- clustering labeling

Usage

    from scipy.io import mmread
    x = mmread(mm_file).tocsr() # Only for sparse matrix

for spherical k-means

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
    
for cluster labeling

    from soyclustering import proportion_keywords
    
    centers = spherical_kmeans.cluster_centers_
    idx2vocab = ['list', 'of', 'str', 'vocab']
    keywords = proportion_keywords(centers, labels, index2word=idx2vocab)
   """,
   install_requires=["scikit-learn>=0.19.1"],
   keywords = ['Document clustering'],
   packages=find_packages(),
)