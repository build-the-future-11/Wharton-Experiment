"""Matched-information zero, ridge and explicit CPU/CUDA XGBoost controls."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from .runtime import write_json
from .training import budget


def baseline(samples, config, arm, seed, directory, deadline, return_validation=False):
    x_train, x_val, x_test = [
        samples.tabular(ix) for ix in (samples.train, samples.validation, samples.test)
    ]
    y_train = samples.y[samples.train] / samples.target_scale
    y_val = samples.y[samples.validation] / samples.target_scale
    budget(deadline)
    if arm == "zero":
        val, test = np.zeros(len(x_val)), np.zeros(len(x_test))
    elif arm == "ridge":
        center, scale = x_train.mean(0), x_train.std(0)
        scale = np.where(scale > 1e-8, scale, 1)
        design = np.c_[np.ones(len(x_train)), (x_train - center) / scale]
        regularizer = np.eye(design.shape[1])
        regularizer[0, 0] = 0
        beta = np.linalg.solve(design.T @ design + regularizer, design.T @ y_train)
        val = np.c_[np.ones(len(x_val)), (x_val - center) / scale] @ beta
        test = np.c_[np.ones(len(x_test)), (x_test - center) / scale] @ beta
        np.savez(
            Path(directory) / "ridge.npz",
            beta=beta,
            center=center,
            scale=scale,
            target_scale=samples.target_scale,
        )
    elif arm == "xgboost":
        import xgboost as xgb

        class Deadline(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                budget(deadline)
                return False

        train = xgb.DMatrix(x_train, label=y_train)
        validation = xgb.DMatrix(x_val, label=y_val)
        model = xgb.train(
            {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "device": config.device,
                "max_depth": 4,
                "eta": 0.05,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "lambda": 5.0,
                "seed": seed,
                "nthread": 1,
                "eval_metric": "rmse",
            },
            train,
            num_boost_round=config.xgb_rounds,
            evals=[(validation, "validation")],
            early_stopping_rounds=config.patience,
            callbacks=[Deadline()],
            verbose_eval=False,
        )
        actual = json.loads(model.save_config())["learner"]["generic_param"]["device"]
        if config.device == "cuda" and not actual.startswith("cuda"):
            raise RuntimeError("XGBoost silently fell back from requested CUDA")
        stop = model.best_iteration + 1
        val = model.predict(validation, iteration_range=(0, stop))
        test = model.predict(xgb.DMatrix(x_test), iteration_range=(0, stop))
        model.save_model(Path(directory) / "xgboost.ubj")
        write_json(
            Path(directory) / "xgboost.json",
            {"best_iteration": model.best_iteration, "device": actual},
        )
    else:
        raise ValueError("unknown baseline")
    # A fixed Student law around the baseline, with validation-only residual scale.
    df = config.model.student_df
    scale = (
        max(float(np.sqrt(np.mean((y_val - val) ** 2) * (df - 2) / df)), 1e-3)
        * samples.target_scale
    )
    write_json(
        Path(directory) / "calibration.json", {"scale": scale, "df": df, "used": "validation_only"}
    )

    def law(prediction, indices):
        mean = np.asarray(prediction, dtype=np.float64) * samples.target_scale
        return {
            "mean": mean,
            "mu": mean[:, None],
            "scale": np.full((len(mean), 1), scale),
            "std": np.full(len(mean), scale * np.sqrt(df / (df - 2))),
            "weights": np.ones((len(mean), 3, 1)),
            "nll": -t.logpdf((samples.y[indices] - mean) / scale, df) + np.log(scale),
        }

    test_law = law(test, samples.test)
    return (test_law, law(val, samples.validation)) if return_validation else test_law


def combine(first, second, weight, y, df):
    """Mixture of distributions, including between-expert variance."""
    from .evaluation import score_law

    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("blend weight must be in [0,1]")
    if weight == 0:
        return second
    if weight == 1:
        return first
    mu = np.concatenate([first["mu"], second["mu"]], 1)
    scale = np.concatenate([first["scale"], second["scale"]], 1)
    weights = np.concatenate([weight * first["weights"], (1 - weight) * second["weights"]], 2)
    mean = (weights[:, 0] * mu).sum(1)
    variance = (weights[:, 0] * (scale**2 * df / (df - 2) + (mu - mean[:, None]) ** 2)).sum(1)
    out = {
        "mu": mu,
        "scale": scale,
        "weights": weights,
        "mean": mean,
        "std": np.sqrt(np.maximum(variance, 1e-12)),
    }
    out["nll"] = score_law(out, y, df)["nll"]
    return out


def hybrid(neural, neural_validation, samples, config, seed, directory, deadline):
    directory = Path(directory)
    directory.mkdir()
    tree, tree_validation = baseline(samples, config, "xgboost", seed, directory, deadline, True)
    weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    # All calibration/selection is on validation. Test outcomes never select the blend.
    losses = [
        float(
            combine(
                neural_validation,
                tree_validation,
                weight,
                samples.y[samples.validation],
                config.model.student_df,
            )["nll"].mean()
        )
        for weight in weights
    ]
    chosen = weights[int(np.argmin(losses))]
    write_json(
        directory / "blend.json",
        {
            "weights": weights,
            "validation_nll": losses,
            "neural_weight": chosen,
            "tie_policy": "first_smallest_neural_weight",
        },
    )
    np.savez(
        directory / "neural_validation.npz", **neural_validation, y=samples.y[samples.validation]
    )
    np.savez(directory / "tree_validation.npz", **tree_validation, y=samples.y[samples.validation])
    return combine(neural, tree, chosen, samples.y[samples.test], config.model.student_df)
