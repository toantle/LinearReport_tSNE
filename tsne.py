# Author: Alexander Fabisch  -- <afabisch@informatik.uni-bremen.de>
# Author: Christopher Moody <chrisemoody@gmail.com>
# Author: Nick Travers <nickt@squareup.com>
# License: BSD 3 clause (C) 2014

# This is the exact and Barnes-Hut t-SNE implementation. There are other
# modifications of the algorithm:
# * Fast Optimization for t-SNE:
#   https://cseweb.ucsd.edu/~lvdmaaten/workshops/nips2010/papers/vandermaaten.pdf
 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy import linalg

MACHINE_EPSILON = np.finfo(np.double).eps

def _binary_search_perplexity(sqdistances, desired_perplexity):
    EPSILON_DBL = 1e-8
    PERPLEXITY_TOLERANCE = 1e-5

    n_steps = 100

    n_samples = sqdistances.shape[0]
    n_neighbors = sqdistances.shape[1]
    using_neighbors = n_neighbors < n_samples
    # Precisions of conditional Gaussian distributions    
    beta_sum = 0.0

    # Use log scale
    desired_entropy = np.log(desired_perplexity)    

    # This array is later used as a 32bit array. It has multiple intermediate
    # floating point additions that benefit from the extra precision
    P = np.zeros((n_samples, n_neighbors), dtype=np.float64)

    for i in range(n_samples):
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0

        # Binary search of precision for i-th conditional distribution
        for l in range(n_steps):
            # Compute current entropy and corresponding probabilities
            # computed just over the nearest neighbors or over all data
            # if we're not using neighbors
            sum_Pi = 0.0
            for j in range(n_neighbors):
                if j != i or using_neighbors:
                    P[i, j] = np.exp(-sqdistances[i, j] * beta)
                    sum_Pi += P[i, j]

            if sum_Pi == 0.0:
                sum_Pi = EPSILON_DBL
            sum_disti_Pi = 0.0

            for j in range(n_neighbors):
                P[i, j] /= sum_Pi
                sum_disti_Pi += sqdistances[i, j] * P[i, j]

            entropy = np.log(sum_Pi) + beta * sum_disti_Pi
            entropy_diff = entropy - desired_entropy

            if np.fabs(entropy_diff) <= PERPLEXITY_TOLERANCE:
                break

            if entropy_diff > 0.0:
                beta_min = beta
                if beta_max == np.inf:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min == -np.inf:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0

        beta_sum += beta

    print("[t-SNE] Mean sigma: %f" % np.mean(np.sqrt(n_samples / beta_sum)))

    return P

def _kl_divergence(params, P, degrees_of_freedom, n_samples, n_components,
                   skip_num_points=0, compute_error=True):
    X_embedded = params.reshape(n_samples, n_components)

    # Q is a heavy-tailed distribution: Student's t-distribution
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Optimization trick below: np.dot(x, y) is faster than
    # np.sum(x * y) because it calls BLAS

    # Objective: C (Kullback-Leibler divergence of P and Q)
    if compute_error:
        kl_divergence = 2.0 * np.dot(
            P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    else:
        kl_divergence = np.nan

    # Gradient: dC/dY
    # pdist always returns double precision distances. Thus we need to take
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(skip_num_points, n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order='K'),
                         X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c

    return kl_divergence, grad

def _gradient_descent(objective, p0, it, n_iter,
                      n_iter_check=1, n_iter_without_progress=300,
                      momentum=0.8, learning_rate=200.0, min_gain=0.01,
                      min_grad_norm=1e-7, args=None, kwargs=None):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = i = it
    steps = []
    for i in range(it, n_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        kwargs['compute_error'] = check_convergence or i == n_iter - 1

        error, grad = objective(p, *args, **kwargs)
        grad_norm = linalg.norm(grad)

        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        if i == 0 or (i + 1) % 50 == 0:
            steps.append([i+1, p.copy().ravel()])

        if check_convergence:            
            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:                
                break
            if grad_norm <= min_grad_norm:                
                break

    return p, error, i, steps

def _joint_probabilities(distances, desired_perplexity):    
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances = distances.astype(np.float32, copy=False)
    conditional_P = _binary_search_perplexity(distances, desired_perplexity)
    P = conditional_P + conditional_P.T
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
    print('Compute joint probability done!')
    return P

def tsne(X=np.array([]), no_dims=2, perplexity=30.0, max_iter=1000, return_steps=False):
    _EXPLORATION_N_ITER=250
    _EARLY_EXAGGERATION=12.0
    n_samples = X.shape[0]

    distances = pairwise_distances(X)
    P = _joint_probabilities(distances, perplexity)

    pca = PCA(n_components=no_dims, svd_solver='randomized', random_state=0)
    X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
    print('Preprocessed with PCA done')
    
    print('Start Gradient descent with early exaggeration')
    degrees_of_freedom = max(no_dims - 1, 1)

    params = X_embedded.ravel()
    opt_args = {
            "it": 0,
            "n_iter_check": 50,
            "min_grad_norm": 1e-7,
            "learning_rate": 200.0,
            "kwargs": dict(skip_num_points=0),
            "args": [P, degrees_of_freedom, n_samples, no_dims],
            "n_iter_without_progress": _EXPLORATION_N_ITER,
            "n_iter": _EXPLORATION_N_ITER,
            "momentum": 0.5,
        }
    obj_func = _kl_divergence
    P *= _EARLY_EXAGGERATION
    params, kl_divergence, it, steps = _gradient_descent(obj_func, params, **opt_args)

    print('Start Gradient descent with no early exaggeration')
    P /= _EARLY_EXAGGERATION
    remaining = max_iter - _EXPLORATION_N_ITER
    if it < _EXPLORATION_N_ITER or remaining > 0:
        opt_args['n_iter'] = max_iter
        opt_args['it'] = it + 1
        opt_args['momentum'] = 0.8
        opt_args['n_iter_without_progress'] = 300
        params, kl_divergence, it, add_steps = _gradient_descent(obj_func, params, **opt_args)
    
    X_embedded = params.reshape(n_samples, no_dims)
    steps = np.vstack((steps, add_steps))
    steps = [[step[0], step[1].reshape(n_samples, no_dims)] for step in steps]
    steps = [[step[0], Y[0], Y[1]] for step in steps for Y in step[1]]

    return X_embedded if not return_steps else (X_embedded, steps)