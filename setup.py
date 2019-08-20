import soyclustering
from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="soyclustering",
    version=soyclustering.__version__,
    author=soyclustering.__author__,
    author_email='soy.lovit@gmail.com',
    url='https://github.com/lovit/clustering4docs',
    description="Python library for document clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["numpy>=1.1"],
    keywords = ['document clustering', 'clustering labeling'],
    packages=find_packages()
)
