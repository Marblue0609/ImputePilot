import json
import os
import time

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand
from django.test import Client, override_settings

from ImputePilot_api.dataset_categories import annotate_benchmark_rows, build_benchmark_category_summary
from ImputePilot_api.views import (
    AdartsService,
    _read_imputed_file,
    _run_imputation_for_algo,
    _run_self_supervised_classification_task,
    _run_self_supervised_forecasting_task,
    _build_eval_mask_from_observed,
    _run_external_baseline_runner,
)

BASELINE_PREDICT_TIMEOUT_SEC = int(os.getenv("ImputePilot_BASELINE_PREDICT_TIMEOUT_SEC", "2400"))


class Command(BaseCommand):
    help = "Run one-time RealWorld downstream evaluation across datasets and baselines."

    def add_arguments(self, parser):
        parser.add_argument("--missing-rate", type=float, default=0.1)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--time-budget", type=int, default=300)
        parser.add_argument("--output", type=str, default="realworld_downstream_eval.json")
        parser.add_argument("--train-baselines", action="store_true", default=True)
        parser.add_argument("--no-train-baselines", action="store_true", default=False)
        parser.add_argument(
            "--dataset",
            action="append",
            default=[],
            help="Run only selected dataset(s). Repeat this arg to pass multiple names.",
        )

    def handle(self, *args, **options):
        missing_rate = float(options["missing_rate"])
        seed = int(options["seed"])
        time_budget = int(options["time_budget"])
        output_name = options["output"]
        train_baselines = options["train_baselines"] and not options["no_train_baselines"]
        requested_datasets = options.get("dataset") or []
        requested_datasets = [str(x).strip() for x in requested_datasets if str(x).strip()]
        requested_set = {x.lower() for x in requested_datasets}

        trained_model = AdartsService.get_trained_model()
        if trained_model is None:
            self.stderr.write("[ERROR] No trained model found. Run ModelRace first.")
            return

        if train_baselines:
            self._ensure_baselines_trained(time_budget=time_budget)

        datasets = AdartsService.load_datasets()
        if requested_set:
            datasets = [d for d in datasets if str(getattr(d, "name", "")).lower() in requested_set]
        if not datasets:
            if requested_set:
                self.stderr.write(f"[ERROR] Requested datasets not found: {sorted(requested_set)}")
            else:
                self.stderr.write("[ERROR] No RealWorld datasets found.")
            return

        rows = []
        methods = ["ImputePilot", "FLAML", "Tune", "AutoFolio", "RAHA"]

        for dataset in datasets:
            ts_df = dataset.load_timeseries(transpose=True)
            ts_df = ts_df.apply(pd.to_numeric, errors="coerce")
            if ts_df.empty:
                continue

            values = ts_df.to_numpy(dtype="float32")
            eval_mask = _build_eval_mask_from_observed(values, seed + abs(hash(dataset.name)) % 10000, missing_rate)
            eval_df = ts_df.copy()
            eval_df.values[eval_mask] = np.nan

            # Fill each column independently (robust to duplicated column names).
            mean_filled = ts_df.apply(
                lambda s: s.fillna(0.0 if pd.isna(s.mean()) else s.mean()),
                axis=0,
            )

            oracle_df = (
                ts_df.interpolate(method="linear", axis=1, limit_direction="both")
                .ffill(axis=1)
                .bfill(axis=1)
                .fillna(0.0)
            )

            X = self._compute_features_for_recommendation(dataset, eval_df, trained_model)
            if X is None:
                self.stderr.write(f"[WARN] Skipping dataset {dataset.name}: failed to extract features.")
                continue

            ImputePilot_algo = self._recommend_ImputePilot_algo(trained_model, X)
            baseline_algos = self._recommend_baseline_algos(X)

            algo_map = {"ImputePilot": ImputePilot_algo}
            algo_map.update(baseline_algos)

            # Cache per-algorithm evaluation in this dataset to avoid repeated expensive
            # imputation/downstream runs when multiple recommenders pick the same algorithm.
            algo_eval_cache = {}

            for method in methods:
                algo = algo_map.get(method)
                if not algo:
                    rows.append(
                        {
                            "dataset": dataset.name,
                            "method": method,
                            "algo": None,
                            "forecasting_rmse": None,
                            "forecasting_improvement": None,
                            "forecasting_n_evaluated": None,
                            "classification_acc": None,
                            "classification_improvement": None,
                            "classification_n_evaluated": None,
                            "status": "skipped",
                            "error": "No algorithm available",
                        }
                    )
                    continue

                algo_key = str(algo).strip().lower()
                cached_eval = algo_eval_cache.get(algo_key)
                if cached_eval is None:
                    imputation_result = _run_imputation_for_algo(eval_df, algo)
                    if imputation_result.get("error"):
                        cached_eval = {
                            "status": "error",
                            "error": imputation_result.get("error"),
                        }
                        algo_eval_cache[algo_key] = cached_eval
                    else:
                        imputed_file = imputation_result.get("imputed_file")
                        if not imputed_file or not os.path.exists(imputed_file):
                            cached_eval = {
                                "status": "error",
                                "error": "Imputed file missing",
                            }
                            algo_eval_cache[algo_key] = cached_eval
                        else:
                            imputed_df = _read_imputed_file(imputed_file)
                            if imputed_df.shape != eval_df.shape and imputed_df.T.shape == eval_df.shape:
                                imputed_df = imputed_df.T

                            forecast_res = _run_self_supervised_forecasting_task(
                                ts_df,
                                imputed_df,
                                mean_filled,
                                eval_mask,
                                oracle_df=oracle_df,
                            )
                            class_res = _run_self_supervised_classification_task(
                                ts_df,
                                imputed_df,
                                mean_filled,
                                oracle_df=oracle_df,
                            )

                            cached_eval = {
                                "status": "success",
                                "forecasting_rmse": forecast_res.get("withImputePilot", forecast_res.get("withAdarts")),
                                "forecasting_improvement": forecast_res.get("improvement"),
                                "forecasting_n_evaluated": forecast_res.get("n_evaluated"),
                                "classification_acc": class_res.get("withImputePilot", class_res.get("withAdarts")),
                                "classification_improvement": class_res.get("improvement"),
                                "classification_n_evaluated": class_res.get("n_evaluated"),
                            }
                            algo_eval_cache[algo_key] = cached_eval

                if cached_eval.get("status") == "error":
                    rows.append(
                        {
                            "dataset": dataset.name,
                            "method": method,
                            "algo": algo,
                            "forecasting_rmse": None,
                            "forecasting_improvement": None,
                            "forecasting_n_evaluated": None,
                            "classification_acc": None,
                            "classification_improvement": None,
                            "classification_n_evaluated": None,
                            "status": "error",
                            "error": cached_eval.get("error"),
                        }
                    )
                    continue

                rows.append(
                    {
                        "dataset": dataset.name,
                        "method": method,
                        "algo": algo,
                        "forecasting_rmse": cached_eval.get("forecasting_rmse"),
                        "forecasting_improvement": cached_eval.get("forecasting_improvement"),
                        "forecasting_n_evaluated": cached_eval.get("forecasting_n_evaluated"),
                        "classification_acc": cached_eval.get("classification_acc"),
                        "classification_improvement": cached_eval.get("classification_improvement"),
                        "classification_n_evaluated": cached_eval.get("classification_n_evaluated"),
                        "status": "success",
                        "error": None,
                    }
                )

        rows = annotate_benchmark_rows(rows)
        category_payload = build_benchmark_category_summary(rows, methods)

        payload = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "missing_rate": missing_rate,
            "seed": seed,
            "methods": methods,
            "rows": category_payload["rows"],
            "categories": category_payload["categories"],
            "category_summary": category_payload["category_summary"],
            "category_stats": category_payload["category_stats"],
        }

        out_dir = AdartsService.get_recommendations_dir()
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, output_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        csv_path = os.path.join(out_dir, output_name.replace(".json", ".csv"))
        try:
            pd.DataFrame(category_payload["rows"]).to_csv(csv_path, index=False)
        except Exception:
            pass

        self.stdout.write(f"[OK] RealWorld downstream evaluation saved to {out_path}")

    def _ensure_baselines_trained(self, time_budget=300):
        client = Client()
        baseline_calls = [
            ("FLAML", "/api/baseline/train_flaml/", {"time_budget": time_budget}),
            ("Tune", "/api/baseline/train_tune/", {"time_budget": time_budget}),
            ("AutoFolio", "/api/baseline/train_autofolio/", {"time_budget": time_budget}),
            ("RAHA", "/api/baseline/train_raha/", {}),
        ]

        with override_settings(ALLOWED_HOSTS=["testserver", "localhost", "127.0.0.1"]):
            for name, endpoint, payload in baseline_calls:
                model = getattr(AdartsService, f"get_{name.lower()}_model", None)
                if callable(model) and model() is not None:
                    continue
                resp = client.post(
                    endpoint,
                    data=json.dumps(payload),
                    content_type="application/json",
                )
                if resp.status_code != 200:
                    self.stderr.write(f"[WARN] {name} training failed: {resp.content[:200]}")
                else:
                    self.stdout.write(f"[INFO] {name} baseline trained.")

    def _compute_features_for_recommendation(self, dataset, timeseries, trained_model):
        try:
            pipelines = trained_model.get("pipelines", [])
            feature_extractors = trained_model.get("feature_extractors", [])
            light_features_mode = os.environ.get("ImputePilot_DASHBOARD_LIGHT_FEATURES", "0") == "1"

            features_name = None
            if pipelines and hasattr(pipelines[0], "rm") and getattr(pipelines[0].rm, "features_name", None) is not None:
                features_name = pipelines[0].rm.features_name

            # Fast-path for RealWorld datasets: reuse previously extracted feature CSVs
            # from the training pipeline to avoid re-running heavy TSFresh/Topological extraction.
            use_cached_features = os.environ.get("ImputePilot_DASHBOARD_USE_CACHED_FEATURES", "1") == "1"
            if use_cached_features and dataset is not None:
                all_cached_features = []
                for fe in feature_extractors:
                    if light_features_mode and fe.__class__.__name__ != "Catch22FeaturesExtractor":
                        continue
                    try:
                        tmp_features = dataset.load_features(fe)
                        tmp_features.set_index("Time Series ID", inplace=True)
                        all_cached_features.append(tmp_features)
                    except Exception:
                        all_cached_features = []
                        break

                if all_cached_features:
                    timeseries_features = pd.concat(all_cached_features, axis=1)
                    nb_timeseries = timeseries_features.shape[0]

                    if features_name is not None:
                        timeseries_features = timeseries_features.reindex(
                            columns=list(features_name), fill_value=0.0
                        )

                    X = timeseries_features.to_numpy().astype("float32")
                    X = np.nan_to_num(
                        X,
                        nan=0.0,
                        posinf=np.finfo(np.float32).max,
                        neginf=np.finfo(np.float32).min,
                    )
                    return X

            timeseries = timeseries.apply(pd.to_numeric, errors="coerce")
            missing_before = int(timeseries.isna().sum().sum())
            if missing_before > 0:
                ts_feature_input = (
                    timeseries.interpolate(method="linear", axis=1, limit_direction="both")
                    .ffill(axis=1)
                    .bfill(axis=1)
                    .fillna(0.0)
                )
            else:
                ts_feature_input = timeseries

            nb_timeseries, timeseries_length = ts_feature_input.shape
            ts_for_extraction = ts_feature_input.T
            all_ts_features = []

            for fe in feature_extractors:
                fe_name = fe.__class__.__name__
                if light_features_mode and fe_name != "Catch22FeaturesExtractor":
                    continue
                args = (
                    (ts_for_extraction, nb_timeseries, timeseries_length)
                    if fe_name == "TSFreshFeaturesExtractor"
                    else (ts_for_extraction,)
                )
                tmp_features = fe.extract_from_timeseries(*args)
                tmp_features.set_index("Time Series ID", inplace=True)
                tmp_features.columns = [
                    col + fe.FEATURES_FILENAMES_ID if col != "Time Series ID" else col
                    for col in tmp_features.columns
                ]
                all_ts_features.append(tmp_features)

            if not all_ts_features:
                return None

            timeseries_features = pd.concat(all_ts_features, axis=1)

            if features_name is not None:
                timeseries_features = timeseries_features.reindex(
                    columns=list(features_name), fill_value=0.0
                )

            X = timeseries_features.to_numpy().astype("float32")
            X = np.nan_to_num(
                X,
                nan=0.0,
                posinf=np.finfo(np.float32).max,
                neginf=np.finfo(np.float32).min,
            )
            return X
        except Exception:
            return None

    def _recommend_ImputePilot_algo(self, trained_model, X):
        pipelines = trained_model.get("pipelines", [])
        voting_results = {}

        for pipe in pipelines:
            try:
                rm = pipe.rm
                best_cv = getattr(rm, "best_cv_trained_pipeline", None)
                prod = getattr(rm, "trained_pipeline_prod", None)
                if best_cv is None and prod is None:
                    continue
                use_prod = prod is not None
                recommendations = rm.get_recommendations(X, use_pipeline_prod=use_prod)
                if recommendations is None or recommendations.empty:
                    continue
                avg_probs = recommendations.mean(axis=0)
                for algo, prob in avg_probs.items():
                    voting_results.setdefault(algo, []).append(float(prob))
            except Exception:
                continue

        if not voting_results:
            return None
        avg_scores = {algo: float(np.mean(probs)) for algo, probs in voting_results.items()}
        sorted_algos = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_algos[0][0] if sorted_algos else None

    def _recommend_baseline_algos(self, X):
        results = {}

        # FLAML
        flaml_model_data = AdartsService.get_flaml_model()
        if flaml_model_data is not None:
            algo = self._predict_baseline("FLAML", flaml_model_data, X)
            if algo:
                results["FLAML"] = algo

        # Tune
        tune_model_data = AdartsService.get_tune_model()
        if tune_model_data is not None:
            algo = self._predict_baseline("TUNE", tune_model_data, X)
            if algo:
                results["Tune"] = algo

        # AutoFolio
        autofolio_model_data = AdartsService.get_autofolio_model()
        if autofolio_model_data is not None:
            algo = self._predict_baseline("AUTOFOLIO", autofolio_model_data, X)
            if algo:
                results["AutoFolio"] = algo

        # RAHA
        raha_model_data = AdartsService.get_raha_model()
        if raha_model_data is not None:
            algo = self._predict_raha_algo(raha_model_data, X)
            if algo:
                results["RAHA"] = algo

        return results

    def _predict_baseline(self, name, model_data, X):
        try:
            if model_data.get("external_runner"):
                ext_res = _run_external_baseline_runner(
                    name,
                    "predict",
                    arrays_dict={"X_infer": X},
                    meta_dict={},
                    timeout_sec=BASELINE_PREDICT_TIMEOUT_SEC,
                )
                if ext_res.get("status") != "success":
                    return None
                return ext_res.get("algo")

            clf = model_data.get("model")
            if clf is None:
                return None
            preds = clf.predict(X)
            if preds is None or len(preds) == 0:
                return None
            values, counts = np.unique(preds, return_counts=True)
            return str(values[np.argmax(counts)])
        except Exception:
            return None

    def _predict_raha_algo(self, raha_model_data, X):
        if raha_model_data.get("external_runner"):
            ext_res = _run_external_baseline_runner(
                "RAHA",
                "predict",
                arrays_dict={"X_infer": X},
                meta_dict={},
                timeout_sec=BASELINE_PREDICT_TIMEOUT_SEC,
            )
            if ext_res.get("status") != "success":
                return None
            return ext_res.get("algo")

        try:
            existing_vectors = raha_model_data["existing_vectors"]
            error_metric = raha_model_data["error_metric"]
            profile_vec = X.mean(axis=0).astype(float)

            def _cosine_dist(a, b):
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na == 0 or nb == 0:
                    return 1.0
                sim = np.dot(a, b) / (na * nb)
                return 1.0 if np.isnan(sim) else 1.0 - sim

            all_techniques = set()
            for _, row in existing_vectors.iterrows():
                if row["Benchmark Results"] is not None:
                    try:
                        all_techniques.update(row["Benchmark Results"].index.tolist())
                    except Exception:
                        pass
            all_techniques = sorted(all_techniques)

            g_max_error = 0.0
            for _, row in existing_vectors.iterrows():
                try:
                    max_e = row["Benchmark Results"][error_metric].max()
                    if max_e > g_max_error:
                        g_max_error = max_e
                except Exception:
                    pass
            if g_max_error == 0:
                g_max_error = 1.0

            scores = {}
            for _, row in existing_vectors.iterrows():
                fv = row["Features Vector"]
                if fv is None:
                    continue
                dist = _cosine_dist(fv.to_numpy().astype(float), profile_vec)
                for technique in all_techniques:
                    try:
                        rmse = row["Benchmark Results"][error_metric].loc[technique] / g_max_error
                        score = dist * rmse
                        if technique not in scores or score < scores[technique]:
                            scores[technique] = score
                    except (KeyError, TypeError):
                        pass

            if not scores:
                return None
            sorted_techniques = sorted(scores.items(), key=lambda x: x[1])
            for tech, _ in sorted_techniques:
                if "cdrec" in tech:
                    return "cdrec"
                return tech
        except Exception:
            return None
