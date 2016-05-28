# Anomaly detection

In this repository you can find a distributed version of an anomaly detection algorithm. The algorithm is based on Dissimilarity Mapping Fitering, developed originally in [S. Morante, J. G. Victores and C. Balaguer, "Automatic demonstration and feature selection for robot learning," Humanoid Robots (Humanoids), 2015 IEEE-RAS 15th International Conference on, Seoul, 2015, pp. 428-433](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7363569&isnumber=7362951).

Basically, it performs a pairwise comparison between all the elements in the dataset using some distance (several defined). Then, it reduces the information to a single dimension and filters by some threshold.

This repository includes two developments:

1. **Signal**: This class allows to input data in several formats (lists, RDD, pandas dataframe, Spark dataframe, etc) and obtain an RDD with special properties:
	- Every row is converted to string or float depending on the dataset (categorical or numerical)
	- Every missing value is imputed with tunneable values
	- Every row is given a unique index

2. **DMF**: This class implements the core idea behind Dissimilarity Mapping Filtering. Each block connects to the next, but it is agnostic to the algorithm used in other blocks. The algorithms available in each block are:
	- Dissimilarity:
		1. Euclidean ('euclidean'): list of data is compared using euclidean distance between elements of the lists
		2. Dynamic Time Warping ('dtw'): it performs a Dynamic Time Warping comparison between the lists
		3. Hamming ('hamming'): more suitable for categorical variables, it performs a Hamming distance calculation between the lists
		
	- Mapping:
		1. Sum ('sum'): it sums all the values of one index into a single value
		
	- Filtering:
		1. Z-score ('zscore'): it performs a normalization of data ((data - mean)/stDev) and filter it using a threshold alpha
		2. Threshold ('zscore'): it filters data based on a threshold alpha
		
		
A notebook is included with an example of use.
