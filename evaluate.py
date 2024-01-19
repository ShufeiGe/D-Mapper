import kmapper as km
import matplotlib.pyplot as plt
import numpy as np
import random
import sklearn
from sklearn import cluster
import networkx as nx
import gudhi
from tqdm import tqdm
from scipy.sparse.csgraph import dijkstra, shortest_path, connected_components
from scipy.stats import ks_2samp
from scipy import stats
from gudhi import bottleneck_distance

#input original data(np.array), mapper output graph
# norm = 0/1: no Normalization/Normalization
# k: Keep K decimal places
def get_SC(data,graph, norm = 0,k = 3): 
    ele_list = []
    r_list = []
    k = 0
    for cluster_list in graph["nodes"].values():
        ele_list = ele_list + cluster_list  #all elements index
        r_list = r_list + [k]*len(cluster_list) #the cluster id
        k = k + 1
    l = len(ele_list)

    result_arry = np.zeros((l,2)) #one for elements index and another for cluster id
    result_arry[:,0] = ele_list
    result_arry[:,1] = r_list
    result_arry = result_arry.astype(int)
    
    new_data = np.zeros((l,data.shape[1])) #to store all data
    
    #find elements by index and store to new_data
    for i in range(0,l):
        new_data[i,:] = data[result_arry[i,0],:]
    score = sklearn.metrics.silhouette_samples(new_data,result_arry[:,1])
    sc = round(np.mean(score),k)
    if norm == 0:
        return sc
    elif norm == 1:
        sc_norm = (round(sc,k)+1)/2
        return sc_norm
    else:
        print("error!")

def find(i, parents):
    if parents[i] == i:
        return i
    else:
        return find(parents[i], parents)

def union(i, j, parents, f):
    if f[i] <= f[j]:
        parents[j] = i
    else:
        parents[i] = j

#compute the extend persist digram, input mapper graph, projected data
def compute_topological_features_for_kmapper(graph, lens, threshold=0.):
    """
    Compute the topological features (connected components, up/down branches, loops) of the 1-skeleton of the cover complex. Connected components and loops are computed with scipy functions, and branches are detected with Union-Find and 0-dimensional persistence of the 1-skeleton.
    Parameters:
        threshold (float): any topological feature whose size is less than this parameter (relative to the first color function) will be discarded.
    Returns:
        dgm (list of (dim,(a,b)) tuples): list of feature characteristics. dim is the topological dimension of the feature (0 for CCs and branches, 1 for loops), a,b are the min and max of the first color function along the feature.
        bnds (list of lists): list of feature points. Each element of this list is the list of point IDs forming the corresponding feature. 
    """
    
    node_list = []
    for s in graph["simplices"]:
        if len(s) == 1:
            node_list.append(s[0])

    node_dic = dict(map(reversed, enumerate(node_list)))

    simplices = []
    for s in graph["simplices"]:
        if len(s) == 1:
            simplices.append([node_dic[s[0]]])
        if len(s) == 2:
            simplices.append([node_dic[s[0]],node_dic[s[1]]])

    st = gudhi.SimplexTree()
    
    for splx in simplices:
        st.insert(splx)

    node_info = {}
    for node in node_list:
        keys = node_dic[node]
        node_info[keys] = np.mean(lens[graph["nodes"][node]])

    num_nodes = st.num_vertices()
    function, namefunc, invnamefunc = {}, {}, {}
    nodeID = 0
    for (s,_) in st.get_skeleton(0):
        namefunc[s[0]] = nodeID
        invnamefunc[nodeID] = s[0]
        function[s[0]] = node_info[s[0]]
        nodeID += 1
    dgm, bnd = [], []
    # connected_components
    A = np.zeros([num_nodes, num_nodes])
    for (splx,_) in st.get_skeleton(1):
        if len(splx) == 2:
            A[namefunc[splx[0]], namefunc[splx[1]]] = 1
            A[namefunc[splx[1]], namefunc[splx[0]]] = 1
    _, ccs = connected_components(A, directed=False)
    for ccID in np.unique(ccs):
        pts = np.argwhere(ccs == ccID).flatten()
        vals = [function[invnamefunc[p]] for p in pts]
        if np.abs(min(vals) - max(vals)) >= threshold:
            dgm.append((0, (min(vals), max(vals))))
            bnd.append([invnamefunc[p] for p in pts])

    # loops
    G = km.adapter.to_networkx(graph)
    try:
        from networkx import cycle_basis
        bndall = cycle_basis(G)
        for pts in bndall:
            vals = [function[node_dic[p]] for p in pts]
            if np.abs(min(vals) - max(vals)) >= threshold:	
                dgm.append((1,(min(vals), max(vals))))
                bnd.append(pts)
    except ImportError:
        print("Networkx not found, loops not computed")
        
    # branches
    for topo_type in ["downbranch", "upbranch"]:

        lfunction = []
        for i in range(num_nodes):
            lfunction.append(function[invnamefunc[i]])

        # upranch is downbranch of opposite function
        if topo_type == "upbranch":
            lfunction = [-f for f in lfunction]

        # sort vertices according to function values and compute inverse function 
        sorted_idxs = np.argsort(np.array(lfunction))
        inv_sorted_idxs = np.zeros(num_nodes)
        for i in range(num_nodes):
            inv_sorted_idxs[sorted_idxs[i]] = i

        # go through all vertices in ascending function order
        persistence_diag, persistence_set, parents, visited = {}, {}, -np.ones(num_nodes, dtype=np.int32), {}
        for i in range(num_nodes):

            current_pt = sorted_idxs[i]
            neighbors = np.ravel(np.argwhere(A[current_pt,:] == 1))
            lower_neighbors = [n for n in neighbors if inv_sorted_idxs[n] <= i] if len(neighbors) > 0 else []

            # no lower neighbors: current point is a local minimum
            if lower_neighbors == []:
                parents[current_pt] = current_pt

            # some lower neighbors exist
            else:

                # find parent pg of lower neighbors with lowest function value
                neigh_parents = [find(n, parents) for n in lower_neighbors]
                pg = neigh_parents[np.argmin([lfunction[n] for n in neigh_parents])]

                # set parent of current point to pg
                parents[current_pt] = pg

                # for each lower neighbor, we will create a persistence diagram point and corresponding set of nodes
                for neighbor in lower_neighbors:

                    # get parent pn
                    pn = find(neighbor, parents)
                    val = lfunction[pn]
                    persistence_set[pn] = []

                    # we will create persistence set only if parent pn is not local minimum pg
                    if pn != pg:
                        # go through all strictly lower nodes with parent pn
                        for v in sorted_idxs[:i]:
                            if find(v, parents) == pn:
                                # if it is already part of another persistence set, continue
                                try:
                                    visited[v]
                                # else, mark visited and include it in current persistence set
                                except KeyError:
                                    visited[v] = True
                                    persistence_set[pn].append(v)

                        # add current point to persistence set
                        persistence_set[pn].append(current_pt)

                        # do union and create persistence point corresponding to persistence set if persistence is sufficiently large
                        if np.abs(lfunction[pn]-lfunction[current_pt]) >= threshold:
                            persistence_diag[pn] = current_pt
                            union(pg, pn, parents, lfunction)

        for key, val in iter(persistence_diag.items()):
            if topo_type == "downbranch":
                dgm.append((0, (lfunction[key],  lfunction[val])))
            elif topo_type == "upbranch":
                dgm.append((0, (-lfunction[val], -lfunction[key])))
            bnd.append([invnamefunc[v] for v in persistence_set[key]])

    bnd = [list(b) for b in bnd]
    return dgm, bnd

#input: all mapper summary
#original data, projected data, dgm from above function, cover
#N: bootstrap number
#type:d/k dmapper or kmapper
def bootstrap_topological_features_for_kmapper(data, lens, dgm, Cover , type, clusterer, N = 100, precompute = False): 
    """
    Use bootstrap to empirically assess stability of the features. This function computes a distribution of bottleneck distances, that can used afterwards to run tests on each topological feature.
    Parameters:
        N (int): number of bootstrap iterations.
    """
    lens = lens.reshape(-1,1)
    num_pts, distribution = len(lens), []

    for bootstrap_id in tqdm(range(N)):

        # Randomly select points
        idxs = np.random.choice(num_pts, size=num_pts, replace=True)
        if precompute == False:
            Xboot = data[idxs,:]
        if precompute == True:
            Xboot = data[idxs][:,idxs]

        f_boot = lens[idxs,:]
        
        if type == 'd':
            mapper = km.D_Mapper(verbose=0)
        if type == 'k':
            mapper = km.KeplerMapper(verbose=0)

        Mboot = mapper.map(
        f_boot,
        Xboot,
        cover = Cover,
        clusterer = clusterer,
        precomputed = precompute)

        # Compute the corresponding persistence diagrams
        dgm_boot, _ = compute_topological_features_for_kmapper(Mboot, f_boot, threshold=0.)

        npts, npts_boot = len(dgm), len(dgm_boot) #compute 0 and 1 dimension homology together
        D1 = np.array([[dgm[pt][1][0], dgm[pt][1][1]] for pt in range(npts)]) 
        D2 = np.array([[dgm_boot[pt][1][0], dgm_boot[pt][1][1]] for pt in range(npts_boot)])
        bottle = bottleneck_distance(D1, D2)
        distribution.append(bottle)
        distri = np.sort(distribution)
    return distri

def get_distance_from_confidence_level_for_kmapper(distribution, alpha=.95):
    """
    Compute the bottleneck distance threshold corresponding to a specific confidence level.
    Parameters:
        alpha (float): confidence level.
    Returns:
        distance value (float); each feature whose size is above this distance is sure at confidence level alpha.
    """
    return distribution[int(alpha*len(distribution))]

def compute_TSR(dgm,d):
    points = np.array([[dgm[pt][1][0], dgm[pt][1][1]] for pt in range(len(dgm))])
    count = 0
    for p in points:
        x = p[0]
        y = p[1]
        if y-x-2*d >= 0 or y-x+2*d <= 0:
            count = count + 1
    rate = count/len(points)
    return rate

def compute_SC_adj(data, lens, graph, Cover ,type , cluster = cluster.DBSCAN(eps=0.5, min_samples=3), N = 100, alpha=.85,  w1 = 0.5, w2 =0.5, precompute = False):
    dgm,_ = compute_topological_features_for_kmapper(graph, lens, threshold=0.)
    dis = bootstrap_topological_features_for_kmapper(data, lens, dgm, Cover ,type ,cluster, N, precompute)
    d = get_distance_from_confidence_level_for_kmapper(dis, alpha)
    rate = compute_TSR(dgm,d)
    SC = get_SC(data,graph,norm=0)
    SC_norm = get_SC(data,graph,norm=1)
    SC_adj = w1*SC_norm + w2*rate
    print("SC:{}".format(SC))
    print("SC_norm:{}".format(SC_norm))
    print("TSR:{}".format(rate))
    print("SC_adj:{}".format(SC_adj))
    return SC_adj