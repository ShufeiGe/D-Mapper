from __future__ import division

try:
    from collections.abc import Iterable
except:
    from collections import Iterable

from sklearn import mixture
from scipy import stats
import math
import warnings
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# TODO: Incorporate @pablodecm's cover API.


__all__ = ["D_Cover", "CubicalCover"]


class D_Cover:
#If you want to expand to other mixed distributions, you need: 
#1. The PDF of the distribution; 
#2. The parameter table of n mixed distributions (k parameters are k+1 rows, and the last row is weights, with n columns). 
    def __init__(self, n_cubes=None, alpha=None, limits=None, n_init = 5, max_iter = 1000,
                 tol = 0.75*1e-3, means_init = None, dis = stats.norm, para_table = None, verbose=0): 
        self.centers_ = None
        self.radius_ = None
        self.inset_ = None
        self.inner_range_ = None
        self.bounds_ = None
        self.di_ = None
        self.interval_table = None
        self.bic = None
        self.prob_table = None 
        self.bayes_n = None
        self.alpha_max = None
        self.tol = tol #therehold of EM algorithm
        self.n_cubes = n_cubes
        self.alpha = alpha
        self.limits = limits
        self.verbose = verbose

        self.n_init = n_init 
        self.max_iter = max_iter
        self.means_init = means_init

        self.dis = dis 
        self.para_table = para_table
        # Check limits can actually be handled and are set appropriately
        assert isinstance(
            self.limits, (list, np.ndarray, type(None))
        ), "limits should either be an array or None"
        if isinstance(self.limits, (list, np.ndarray)):
            self.limits = np.array(self.limits)
            assert self.limits.shape[1] == 2, "limits should be (n_dim,2) in shape"

    def __repr__(self):
        return "Cover(n_cubes=%s, alpha=%s, limits=%s, verbose=%s)" % (
            self.n_cubes,
            self.alpha,
            self.limits,
            self.verbose,
        )

    def _compute_bounds(self, data):

        # If self.limits is array-like
        if isinstance(self.limits, np.ndarray):
            # limits_array is used so we can change the values of self.limits from None to the min/max
            limits_array = np.zeros(self.limits.shape)
            limits_array[:, 0] = np.min(data, axis=0)
            limits_array[:, 1] = np.max(data, axis=0)
            limits_array[self.limits != np.float("inf")] = 0
            self.limits[self.limits == np.float("inf")] = 0
            bounds_arr = self.limits + limits_array
            """ bounds_arr[i,j] = self.limits[i,j] if self.limits[i,j] == inf
                bounds_arr[i,j] = max/min(data[i]) if self.limits == inf """
            bounds = (bounds_arr[:, 0], bounds_arr[:, 1])

            # Check new bounds are actually sensible - do they cover the range of values in the dataset?
            if not (
                (np.min(data, axis=0) >= bounds_arr[:, 0]).all()
                or (np.max(data, axis=0) <= bounds_arr[:, 1]).all()
            ):
                warnings.warn(
                    "The limits given do not cover the entire range of the lens functions\n"
                    + "Actual Minima: %s\tInput Minima: %s\n"
                    % (np.min(data, axis=0), bounds_arr[:, 0])
                    + "Actual Maxima: %s\tInput Maxima: %s\n"
                    % (np.max(data, axis=0), bounds_arr[:, 1])
                )

        else:  # It must be None, as we checked to see if it is array-like or None in __init__
            bounds = (np.min(data, axis=0), np.max(data, axis=0))

        return bounds

    def fit(self, data): #do some change here
        """Fit a cover on the data. This method constructs centers and radii in each dimension given the `perc_overlap` and `n_cube`.

        Parameters
        ============

        data: array-like
            Data to apply the cover to. Warning: First column must be an index column.

        Returns
        ========

        centers: list of arrays
            A list of centers for each cube

        """

        # TODO: support indexing into any columns
        di = np.array(range(1, data.shape[1]))
        indexless_data = data[:, di]
        n_dims = indexless_data.shape[1]  #the nuber of columns
        
        # support different values along each dimension

        ## -- is a list, needs to be array
        ## -- is a singleton, needs repeating
        if isinstance(self.n_cubes, Iterable):
            n_cubes = np.array(self.n_cubes)
            assert (
                len(n_cubes) == n_dims
            ), "Custom cubes in each dimension must match number of dimensions"
        else:
            n_cubes = np.repeat(self.n_cubes, n_dims)

        if isinstance(self.alpha, Iterable):
            alpha = np.array(self.alpha)
            assert (
                len(alpha) == n_dims
            ), "Custom cubes in each dimension must match number of dimensions"
        else:
            alpha = np.repeat(self.alpha, n_dims)


        # create and fit GMM
        dis = self.dis
        if isinstance(dis, stats.norm.__class__):
        # if n_cubes == None, try to use dirichlet process find n
            if self.n_cubes == None:
                from sklearn.mixture import BayesianGaussianMixture
                bgmm = BayesianGaussianMixture(
                    n_components=30, 
                    covariance_type='full',
                    weight_concentration_prior=0.01,
                    weight_concentration_prior_type='dirichlet_process',
                    max_iter = 1000,
                    n_init = 1
                )
                
                bgmm.n_iter_ = 1000
                bgmm.fit(indexless_data)         
                weights = bgmm.weights_

                def get_n(weights, threshold):

                    sorted_weights = sorted(weights, reverse=True) 

                    sum_until_threshold = 0
                    n = 0
                    for w in sorted_weights:
                        n = n+1
                        sum_until_threshold += w
                        if sum_until_threshold > threshold:
                            break
                    return n
                bayes_n = get_n(weights,0.95)
                self.bayes_n = bayes_n
                self.n_cubes = bayes_n 
                print("bayes_n:{}".format(bayes_n))
            
        #fit GMM
            gmm = mixture.GaussianMixture(n_components=self.n_cubes, covariance_type="full",means_init = self.means_init, tol=self.tol,verbose_interval=1)
            gmm.n_init = self.n_init
            gmm.max_iter = self.max_iter
            gmm.fit(indexless_data) 
            bic = gmm.bic(indexless_data)
            prob_table = gmm.predict_proba(indexless_data)
    #compute radius by alpha
        
        if isinstance(dis, stats.norm.__class__):
            G_para_table = np.ones((3,self.n_cubes))
            for j in range(0,self.n_cubes):
                G_para_table[0][j] = gmm.means_[j][0]
                G_para_table[1][j] = np.sqrt(gmm.covariances_[j][0][0]) 
                G_para_table[2][j] = gmm.weights_[j]
            G_para_table = G_para_table[:,np.argsort(G_para_table[0])]

            #find alpha_max for GMM
            area_list = []
            for j in range(0,self.n_cubes-1):
                mu1 = G_para_table[0][j]
                mu2 = G_para_table[0][j+1]
                delta_1 = G_para_table[1][j]
                delta_2 = G_para_table[1][j+1]

                def eq_cdf(x):
                    return dis.cdf(x, mu1, delta_1) - dis.cdf(2*mu1-x, mu1, delta_1) - dis.cdf(2*mu2 - x, mu2, delta_2) + dis.cdf(x, mu2, delta_2)

                from scipy.optimize import fsolve
                x = fsolve(eq_cdf, (mu1+mu2)/2)
                x = x[0]
                s1 = dis.cdf(x, loc = mu1, scale = delta_1) - dis.cdf(2*mu1 - x, loc = mu1, scale = delta_1 )
                #s2 = dis.cdf(2*mu2 - x, loc = mu2, scale = delta_2) - dis.cdf(x, loc = mu2, scale = delta_2 )
                if x < mu1 or x >mu2:
                    print('wrong point in alpha_max finding!')
                if x != mu1 and x != mu2 and s1 <= 0.995:
                    area_list.append(s1)

            #a check
            if area_list == []:
                area_list.append(0)
                print("can't find alpha_max")
            max_area = max(area_list)

            def truncate(num, n):
                integer = int(num * (10**n))/(10**n)
                return float(integer)
            
            #print("alpha_max:{}".format(truncate(1-max_area,2)))
            self.alpha_max = truncate(1-max_area,2)
            if self.alpha == None:
                self.alpha = self.alpha_max*0.9 #if alpha = None this will be the alpha
            self.para_table = G_para_table
        else:
            # if other distribution we do not provide alpha_max 
            G_para_table = self.para_table




    #Calculate the interval based on quantiles
        interval_table = np.zeros((2,self.n_cubes))

        for i in range(self.n_cubes):
            a_args = list(G_para_table[:,i][:-1])
            b_args = list(G_para_table[:,i][:-1])

            a = dis.ppf(self.alpha/2, *a_args)
            b = dis.ppf(1-self.alpha/2, *b_args)
            interval_table[0,i] = a
            interval_table[1,i] = b
        
        bounds = self._compute_bounds(indexless_data)
        ranges = bounds[1] - bounds[0]

        interval_table[0][0] = bounds[0]
        interval_table[1][-1] = bounds[1]
        
        # (n-1)/n |range|
        inner_range = ((self.n_cubes - 1) / self.n_cubes) * ranges
        inset = (ranges - inner_range) / 2


    #Calculate the radius and center point
        centers = []
        radius = []
        for j in range(0,self.n_cubes):
            r = 0.5*(interval_table[1][j]-interval_table[0][j])
            radius.append(r)        #compute radius
            centers.append(interval_table[0][j]+r)  #compute centers

        self.centers_ = centers
        self.radius_ = radius
        self.inset_ = inset
        self.inner_range_ = inner_range
        self.bounds_ = bounds
        self.di_ = di
        self.interval_table = interval_table
        self.bic = bic
        self.prob_table = prob_table
        self.means_init = self.para_table[0,:].reshape(self.n_cubes,1)

        if self.verbose > 0:
            if isinstance(dis, stats.norm.__class__):
            #draw pdf from fitted GMM

                def normpdf(x, mu, sigma):
                    pdf = np.exp(-((x - mu)**2)/(2*(sigma))) / (np.sqrt(2*np.pi*sigma))
                    return pdf
                
                def GMMpdf(x,mu_,sigma_,n_components,weights):
                    sum = 0
                    for i in range(0,n_components):
                        sum = sum + weights[i]*normpdf(x, mu_[i][0], sigma_[i][0][0])
                    return sum

                x = np.linspace(min(indexless_data), max(indexless_data), 1000)
                
                for i in range(0,self.n_cubes):
                    y = normpdf(x, gmm.means_[i][0], gmm.covariances_[i][0][0])
                    plt.plot(x, y)

                y_G = GMMpdf(x,gmm.means_,gmm.covariances_,self.n_cubes,gmm.weights_)
                plt.plot(x, y_G)
                plt.show()
            
            print(
                " - Cover - centers: %s\ninner_range: %s\nradius: %s"
                % (self.centers_, self.inner_range_, self.radius_)
            )
            print("bic:{}".format(self.bic))
            print("alpha_max:{}".format(self.alpha_max))
            #Check if there are any intersections between adjacent intervals
            for j in range(0,self.n_cubes-1): 
                if interval_table[1][j] < interval_table[0][j+1]:
                    #interval_table[0][j+1] = interval_table[1][j]
                    print("{} and {} have no intersection".format(j,j+1))
                    for d in range(indexless_data.shape[0]):
                        if interval_table[1][j] < indexless_data[d] < interval_table[0][j+1]:
                            print("data {} not in the interval!".format(d))
        return centers #centers in GMM is means

    def transform_single(self, data, center, i=0):
        """Compute entries of `data` in hypercube centered at `center`

        Parameters
        ===========

        data: array-like
            Data to find in entries in cube. Warning: first column must be index column.
        center: array-like
            Center points for the cube. Cube is found as all data in `[center-self.radius_, center+self.radius_]`
        i: int, default 0
            Optional counter to aid in verbose debugging.
        """

        lowerbounds, upperbounds = center - self.radius_[i], center + self.radius_[i]

        # Slice the hypercube
        entries = (data[:, self.di_] >= lowerbounds) & (
            data[:, self.di_] <= upperbounds
        )
        hypercube = data[np.invert(np.any(entries == False, axis=1))]

        if self.verbose > 1:
            print(
                "There are %s points in cube %s/%s"
                % (hypercube.shape[0], i + 1, len(self.centers_))
            )

        return hypercube

    def transform(self, data, centers=None):
        """Find entries of all hypercubes. If `centers=None`, then use `self.centers_` as computed in `self.fit`.

            Empty hypercubes are removed from the result

        Parameters
        ===========

        data: array-like
            Data to find in entries in cube. Warning: first column must be index column.
        centers: list of array-like
            Center points for all cubes as returned by `self.fit`. Default is to use `self.centers_`.

        Returns
        =========
        hypercubes: list of array-like
            list of entries in each hypercube in `data`.

        """

        centers = centers or self.centers_
        hypercubes = [
            self.transform_single(data, cube, i) for i, cube in enumerate(centers)
        ]

        # Clean out any empty cubes (common in high dimensions)
        hypercubes = [cube for cube in hypercubes if len(cube)]
        return hypercubes

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def find(self, data_point):
        """Finds the hypercubes that contain the given data point.

        Parameters
        ===========

        data_point: array-like
            The data point to locate.

        Returns
        =========
        cube_ids: list of int
            list of hypercube indices, empty if the data point is outside the cover.

        """
        cube_ids = []
        for i, center in enumerate(self.centers_):
            lower_bounds, upper_bounds = center - self.radius_[i], center + self.radius_[i]
            if np.all(data_point >= lower_bounds) and np.all(
                data_point <= upper_bounds
            ):
                cube_ids.append(i)
        return cube_ids


class CubicalCover(D_Cover):
    """
    Explicit definition of a cubical cover as the default behavior of the cover class. This is currently identical to the default cover class.
    """

    pass