from dataclasses import replace
import itertools
import warnings
import numpy as np
from .codec import compress


def pareto_indices(values):
    y = np.asarray(values, dtype=float)
    if y.ndim != 2 or y.shape[1] != 2 or not np.isfinite(y).all():
        raise ValueError("Expected finite (n, 2) minimization objectives")
    return [i for i, point in enumerate(y)
            if not np.any(np.all(y <= point, axis=1) & np.any(y < point, axis=1))]


def hypervolume(values, reference):
    points = np.asarray(values, dtype=float).reshape(-1, 2)
    ref = np.asarray(reference, dtype=float)
    if ref.shape != (2,) or not np.isfinite(ref).all() or not np.isfinite(points).all():
        raise ValueError("Invalid hypervolume inputs")
    points = points[np.all(points < ref, axis=1)]
    if not len(points):
        return 0.0
    points = points[np.argsort(points[:, 0], kind="stable")]
    area, best_y = 0.0, ref[1]
    for x, y in points:
        if y < best_y:
            area += (ref[0]-x) * (best_y-y)
            best_y = y
    return float(area)


def configuration_grid(base, block_sizes=(8, 16, 32), seed_bits=(6, 8),
                       group_sizes=(8, 32), thresholds=(0.8, 0.9, 0.95)):
    return [replace(base, block_size=b, seed_bits=s, group_size=g,
                    max_bases=b, energy_threshold=r, fixed_bases=None, lfsr_mask=None,
                    seed_budget=None if base.seed_budget is None else min(base.seed_budget, 2**s-1))
            for b, s, g, r in itertools.product(block_sizes, seed_bits, group_sizes, thresholds)]


def search(weight, configs, trials=12, initial=4, mc_samples=128, random_seed=0,
           target_bpw=None, progress=None):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    from sklearn.exceptions import ConvergenceWarning
    if trials < 1 or initial < 1 or mc_samples < 1 or not configs:
        raise ValueError("Search budget and sample counts must be positive")
    rng = np.random.default_rng(random_seed)
    features = np.array([[c.block_size, c.seed_bits, c.group_size, c.energy_threshold] for c in configs], float)
    x = (features-features.min(0)) / np.maximum(np.ptp(features, axis=0), 1e-12)
    observations, used = [], []
    initial_order = rng.permutation(len(configs)).tolist()
    for iteration in range(min(trials, len(configs))):
        if iteration < min(initial, trials):
            index = initial_order[iteration]
        else:
            y = np.array([[o["mse"], o["logical_bits_per_weight"]] for o in observations])
            # A common affine transform is used for observations, posteriors, and HV.
            low = y.min(0)
            span = np.maximum(np.ptp(y, axis=0), np.maximum(np.abs(y.mean(0))*0.1, 1e-12))
            normalized = (y-low)/span
            reference = normalized.max(0) + 1.0
            remaining = [j for j in range(len(configs)) if j not in used]
            means, stds = [], []
            for m in range(2):
                gp = GaussianProcessRegressor(kernel=ConstantKernel(1., (1e-3, 1e3))*Matern(
                    length_scale=np.ones(4), length_scale_bounds=(1e-2, 1e2), nu=2.5),
                    alpha=1e-6, normalize_y=True, random_state=random_seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    gp.fit(x[used], normalized[:, m])
                mu, sigma = gp.predict(x[remaining], return_std=True)
                means.append(mu)
                stds.append(sigma)
            mu, sigma = np.stack(means, 1), np.stack(stds, 1)
            # Common random numbers reduce acquisition-comparison noise.
            noise = rng.standard_normal((mc_samples, 2))
            base_hv = hypervolume(normalized, reference)
            scores = [np.mean([max(0., hypervolume(np.vstack([normalized, sample]), reference)-base_hv)
                               for sample in mu[j]+sigma[j]*noise]) for j in range(len(remaining))]
            index = remaining[int(np.argmax(scores))]
        result = compress(weight, configs[index])
        used.append(index)
        observations.append({"config": configs[index].to_dict(), **result.metrics})
        if progress:
            progress(iteration+1, observations[-1])
    y = np.array([[o["mse"], o["logical_bits_per_weight"]] for o in observations])
    front = pareto_indices(y)
    feasible = [i for i in front if target_bpw is None or y[i, 1] <= target_bpw]
    if not feasible:
        chosen = None
    elif target_bpw is not None:
        chosen = min(feasible, key=lambda i: y[i, 0])
    else:
        normalized = (y-y.min(0)) / np.maximum(np.ptp(y, axis=0), 1e-12)
        chosen = min(feasible, key=lambda i: float(np.linalg.norm(normalized[i])))
    return {"algorithm": "independent Matern-5/2 GP + Monte Carlo EHVI",
            "objectives": ["per_weight_mse", "logical_bits_per_weight_including_counts"],
            "random_seed": random_seed, "mc_samples": mc_samples,
            "observations": observations, "pareto_indices": front, "selected_index": chosen,
            "selected_config": observations[chosen]["config"] if chosen is not None else None,
            "target_bpw": target_bpw, "target_feasible": bool(feasible)}
