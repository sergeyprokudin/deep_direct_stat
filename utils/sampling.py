import numpy as np


def sample_multiple_gauassians_np(mu, std, n_samples=10):
    """Sample points from multiple multivariate gaussian distributions

    Parameters
    ----------

    mu: numpy array of shape [n_points, n_dims]
        mean values of multiple multivariate gaussians
    std: numpy array of shape [n_points, n_dims]
        stdev values of multiple multivariate gaussians
    n_samples: int
        number of samples to draw from each distribution

    Returns
    -------

    samples: numpy array of shape [n_points, n_samples, n_dims]
        samples from each gaussian
    """
    n_points, n_dims = mu.shape

    eps = np.random.normal(size=[n_points, n_samples, n_dims])

    mu_tiled = np.tile(mu.reshape([n_points, 1, n_dims]), [1, n_samples, 1])
    std_tiled = np.tile(std.reshape([n_points, 1, n_dims]), [1, n_samples, 1])

    samples = mu_tiled + eps*std_tiled

    return samples


def sample_von_mises_mixture(mus_rad, kappas, component_probs, n_samples=100):
    """ Sample from Von-Mises mixture model

    Parameters
    ----------

    mus_rad: array of shape [n_components]
        mean values for each distribution in mixture (in radians)
    kappas: array of shape [n_components]
        kappa values for each distribution in mixture
    component_probs: array of shape [n_components]
        probability of a each component
    n_samples: int
        number of samples to draw

    Returns
    -------

    samples: array of shape [n_samples]
        sampled angles (in radians)
    """

    component_probs = np.clip(component_probs-0.0001, 0.0, 0.9999)
    sample_comps_id = np.nonzero(np.random.multinomial(1, component_probs, n_samples))[1]
    sample_mus = mus_rad[sample_comps_id]
    sample_kappas = kappas[sample_comps_id]

    samples = np.concatenate([np.random.vonmises(sample_mus[sid], sample_kappas[sid]) for sid in range(0, n_samples)])

    return samples


def sample_von_mises_mixture_multi(mus_rad, kappas, component_probs, n_samples=100):
    """ Sample from multiple Von-Mises mixture model

    Parameters
    ----------

    mus_rad: array of shape [n_mixtures, n_components]
        mean values for each distribution in mixture (in radians)
    kappas: array of shape [n_mixtures, n_components, n_dims]
        kappa values for each distribution in mixture
    component_probs: array of shape [n_mixtures, n_components]
        probability of a each component
    n_samples: int
        number of samples to draw

    Returns
    -------

    samples: array of shape [n_mixtures, n_samples]
        sampled angles (in radians)
    """

    samples = [np.reshape(sample_von_mises_mixture(mus_rad[fid], kappas[fid], component_probs[fid],
                                                   n_samples=n_samples), [1, -1])
               for fid in range(0, len(component_probs))]

    return np.concatenate(samples, axis=0)