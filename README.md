# D-Mapper
D-Mapper is a distribution-guided Mapper algorithm, it implements the distribution-guided Mapper algorithm described in 

[Yuyang Tao, Shufei Ge, A distribution-guided Mapper algorithm, 2024.](https://arxiv.org/abs/2401.12237)


The D-Mapper can provide flexible interval construction base on the mixture distribution of projected data.  

This algorithm is programmed as an extension to the KeplerMapper ([Background — KeplerMapper 2.0.1 documentation (scikit-tda.org)](https://kepler-mapper.scikit-tda.org/en/latest/theory.html)) . We provide three extra modules: dcover, dmapper and evaluation. 

# Citation
If you use D-Mapper in your research, please cite the following mannuscript:

Yuyang Tao, Shufei Ge. A distribution-guided Mapper algorithm, 2024.


# Installation
* Method1. Add these three modules into the [kmapper package](https://kepler-mapper.scikit-tda.org/en/latest/index.html)  and import these three modules in \__init\__.py.
* Method2. Download the [kmapper](https://kepler-mapper.scikit-tda.org/en/latest/index.html) directory directly and put it into your import path.

## kmapper.D_Cover
*class* D_Cover(n_cubes=None, alpha=None, limits=None, n_init = 5, max_iter = 1000, tol = 0.75*1e-3, means_init = None, dis = stats.norm, para_table = None, verbose=0)
### Parameters
* **n_cubes: int, default = None**
	Number of hypercubes along each dimension.
	
* **alpha: float, default = None** 
	The symmetrical $\alpha$ quantile of each distribution. This parameter controls the overlap of intervals.
	
* **limits: (_Numpy Array_ _(n_dim,2)_)**
	Same as kmapper, if limits == None, the limits are defined by the maximum and minimum value of the lens for all dimensions.
	
* **n_init: int, default=5**
	The number of initializations to perform in GMM. The best results are kept.
	
* **max_iter:  int, default=1000**
	The number of EM iterations to perform in GMM.
	
* **tol: float, default=0.75e-3**
	The convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold. Sometimes the instability of D-Mapper is caused by this parameter, we suggest to decrease this parameter to get more stable results. 
	
* **means_init: array-like of shape (n_components, n_features), default=None**
	The user-provided initial means, If it is None, means are initialized using the `init_params` method in  sklearn.mixture.GaussianMixture.
	
* **dis: stats, default=stats.norm**
	 The distribution used for constructing mixture model. This should be one of distributions in stats package. The default is normal distribution.
	 
* **para_table: array-like of shape (n_distribution parameters+1, n_distributions), default=None**
	This parameter is only used for none GMM case. The user should provide parameters for each distributions, and the weights should put into the last row. For GMM, this table could be calculated by EM algorithm.

### Methods
All methods in D_Cover can be used the same as in the kmapper.

| Method | Description  |
|---|---|
|[`__init__`](https://kepler-mapper.scikit-tda.org/en/latest/reference/stubs/kmapper.Cover.html#kmapper.Cover.__init__ "kmapper.Cover.__init__")([n_cubes, perc_overlap, limits, ...])||
|[`find`](https://kepler-mapper.scikit-tda.org/en/latest/reference/stubs/kmapper.Cover.html#kmapper.Cover.find "kmapper.Cover.find")(data_point)|Finds the hypercubes that contain the given data point.|
|[`fit`](https://kepler-mapper.scikit-tda.org/en/latest/reference/stubs/kmapper.Cover.html#kmapper.Cover.fit "kmapper.Cover.fit")(data)|Fit a cover on the data.|
|`fit_transform`(data)||
|[`transform`](https://kepler-mapper.scikit-tda.org/en/latest/reference/stubs/kmapper.Cover.html#kmapper.Cover.transform "kmapper.Cover.transform")(data[, centers])|Find entries of all hypercubes.|
|[`transform_single`](https://kepler-mapper.scikit-tda.org/en/latest/reference/stubs/kmapper.Cover.html#kmapper.Cover.transform_single "kmapper.Cover.transform_single")(data, center[, i])|Compute entries of data in hypercube centered at center|

## kmapper.D_Mapper
This class is the D_Mapper implemented as an extention to the KeplerMapper. The major difference is that the cover in D_Mapper must be the D_Cover class.

_class_ kmapper.D_Mapper(_verbose=0_)
### Parameters

- **lens: Numpy Array, default = None**
	Lower dimensional representation of data. In general, it will be output of fit_transform.
    
- **X: Numpy Array, default = None** 
	Original data or data to run clustering on. If None, then use lens as default. X can be a SciPy sparse matrix.
    
- **clusterer: Scikit-learn API, default = DBSCAN**  
	Scikit-learn API compatible clustering algorithm. Must provide fit and predict.
    
- **cover: D_Cover, default = None**
	Distribution guided cover scheme for lens. Must be D_Cover class.
    
- **nerve: _kmapper.Nerve_, default = None**
	Nerve builder implementing __call__(nodes) API.
    
- **precomputed: _Boolean_, default = False** 
	Whether the data to be clustered is a precomputed distance matrix. Same as KeplerMapper.
    
- **remove_duplicate_nodes: _Boolean_, default = False** 
	Remove duplicate nodes before edges are determined. A node is considered to be duplicate if it has exactly the same set of points as another node.

### Methods
All methods in D_Mapper can be used the same as in the kmapper.

| Method  |  Description |
|---|---|
|[`__init__`](https://kepler-mapper.scikit-tda.org/en/latest/reference/stubs/kmapper.KeplerMapper.html#kmapper.KeplerMapper.__init__ "kmapper.KeplerMapper.__init__")([verbose])|Constructor for KeplerMapper class.|
|[`data_from_cluster_id`](https://kepler-mapper.scikit-tda.org/en/latest/reference/stubs/kmapper.KeplerMapper.html#kmapper.KeplerMapper.data_from_cluster_id "kmapper.KeplerMapper.data_from_cluster_id")(cluster_id, graph, data)|Return the original data of each cluster member for a given cluster ID|
|[`fit_transform`](https://kepler-mapper.scikit-tda.org/en/latest/reference/stubs/kmapper.KeplerMapper.html#kmapper.KeplerMapper.fit_transform "kmapper.KeplerMapper.fit_transform")(X[, projection, scaler, ...])|Same as .project() but accept lists for arguments so you can chain.|
|[`map`](https://kepler-mapper.scikit-tda.org/en/latest/reference/stubs/kmapper.KeplerMapper.html#kmapper.KeplerMapper.map "kmapper.KeplerMapper.map")(lens[, X, clusterer, cover, nerve, ...])|Apply Mapper algorithm on this projection and build a simplicial complex.|
|[`project`](https://kepler-mapper.scikit-tda.org/en/latest/reference/stubs/kmapper.KeplerMapper.html#kmapper.KeplerMapper.project "kmapper.KeplerMapper.project")(X[, projection, scaler, distance_matrix])|Create the projection/lens from a dataset.|
|[`visualize`](https://kepler-mapper.scikit-tda.org/en/latest/reference/stubs/kmapper.KeplerMapper.html#kmapper.KeplerMapper.visualize "kmapper.KeplerMapper.visualize")(graph[, color_values, ...])|Generate a visualization of the simplicial complex mapper output.|

## kmapper.evaluate.compute_SC_adj(data, lens, graph, Cover ,type , cluster = cluster.DBSCAN(eps=0.5, min_samples=3), N = 100, alpha=.85,  w1 = 0.5, w2 =0.5, precompute = False)

This is a method to compute adjusted silhouette coefficient.

## Parameters
* **data: Numpy Array, default = None**
	The original input data.
	
* **lens: Numpy Array, default = None**
	The projected data.
	
* **graph: dict, default = None**
	The output of a Mapper graph. The return of the D_Mapper.map() or Mapper.map() method, is the Mapper graph that use want to evaluate.
	
* **Cover: D_Cover or Cover, default = None**
	The Cover to construct the mapper graph. It can be passed directly by the original cover. 
	
* **type: "d" or "k", default = None**
	The type of Mapper user want to evaluate. "d" for D_Mapper, "k" for KeplerMapper.
	
* **cluster: Scikit-learn API, default: DBSCAN(eps=0.5, min_samples=3)**
	The clustering method used in Mapper.
	
* **N: int, default = 100**
	The iteration number of boostrap.
	
* **alpha: float, default = 0.85**
	The confidence level of bottleneck distance.
	
* **w1: float, default = 0.5**
	The weight of silhouette coefficient.
	
* **w2: float, default = 0.5**
	The weight of TSR.
	
* **precompute: _Boolean_, default = False**
	Tell Mapper whether the data that you are clustering on is a precomputed distance matrix. Same as KeplerMapper.
# Return
sc_adj: int
The adjusted silhouette coefficient.


# LICENSES
D- Mapper v1.0. Copyright (c) 2024. Yuyang Tao and Shufei Ge.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS ALGORITHM IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Help
Please send bugs, feature requests and comments to taoyy2022@shanghaitech.edu.cn.
