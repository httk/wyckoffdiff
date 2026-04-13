import torch


def frechet_distance(mu_1, sigma_1, mu_2, sigma_2):
    assert len(mu_1.shape) == len(mu_2.shape) == 1
    assert len(sigma_1.shape) == len(sigma_2.shape) == 2
    assert mu_1.shape == mu_2.shape
    assert sigma_1.shape == sigma_2.shape
    assert sigma_1.shape[0] == sigma_1.shape[1] == mu_1.shape[0]
    mean_diff_norm = (mu_1 - mu_2).square().sum()
    cov_traces = sigma_1.trace() + sigma_2.trace()
    cov_product_trace = torch.linalg.eigvals(sigma_1 @ sigma_2).sqrt().real.sum()
    return mean_diff_norm + cov_traces - 2 * cov_product_trace


def frechet_distance_from_embeddings(feats_1, feats_2):
    assert len(feats_1.shape) == len(feats_2.shape) == 2
    assert feats_1.shape[-1] == feats_2.shape[-1]
    # for improved stability, use double precision
    feats_1 = feats_1.double()
    feats_2 = feats_2.double()
    mu_1 = torch.mean(feats_1, dim=0)
    mu_2 = torch.mean(feats_2, dim=0)
    cov_1 = torch.cov(feats_1.T)
    cov_2 = torch.cov(feats_2.T)

    return frechet_distance(mu_1, cov_1, mu_2, cov_2)
