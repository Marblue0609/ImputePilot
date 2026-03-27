from collections import defaultdict


CATEGORY_ORDER = ["Climate", "Water", "Food/Spectro", "Motion", "Other"]


def get_dataset_category(dataset_name):
    dataset = str(dataset_name or "").strip()
    dataset_lower = dataset.lower()

    if dataset_lower.startswith("climate_") or dataset_lower in {"meteo", "temp", "airq"}:
        return "Climate"
    if dataset_lower == "chlorine" or dataset_lower.startswith("bafu_"):
        return "Water"
    if dataset_lower in {"ham", "herring", "fish", "meat"}:
        return "Food/Spectro"
    if dataset.startswith("GunPoint") or dataset_lower == "haptics":
        return "Motion"
    return "Other"


def order_categories(categories):
    unique = [str(category) for category in categories if str(category).strip()]
    seen = set()
    deduped = []
    for category in unique:
        if category not in seen:
            deduped.append(category)
            seen.add(category)
    preferred = [category for category in CATEGORY_ORDER if category in seen]
    remaining = sorted(category for category in deduped if category not in CATEGORY_ORDER)
    return preferred + remaining


def annotate_benchmark_rows(rows):
    annotated_rows = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        row_copy = dict(row)
        row_copy["category"] = row_copy.get("category") or get_dataset_category(row_copy.get("dataset"))
        annotated_rows.append(row_copy)
    return annotated_rows


def _to_finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _resolve_dataset_weight(dataset_name, dataset_weight_map):
    if not dataset_weight_map:
        return None
    dataset = str(dataset_name or "").strip()
    if not dataset:
        return None
    direct = dataset_weight_map.get(dataset)
    if direct is not None:
        return direct
    return dataset_weight_map.get(dataset.lower())


def _resolve_weight(row, metric_prefix, dataset_weight_map):
    metric_weight = row.get(f"{metric_prefix}_n_evaluated")
    metric_weight = _to_finite_float(metric_weight)
    if metric_weight is not None and metric_weight > 0:
        return metric_weight

    dataset_weight = _resolve_dataset_weight(row.get("dataset"), dataset_weight_map)
    dataset_weight = _to_finite_float(dataset_weight)
    if dataset_weight is not None and dataset_weight > 0:
        return dataset_weight

    return 1.0


def _weighted_average(values_and_weights):
    numerator = 0.0
    denominator = 0.0
    for value, weight in values_and_weights:
        value = _to_finite_float(value)
        weight = _to_finite_float(weight)
        if value is None or weight is None or weight <= 0:
            continue
        numerator += value * weight
        denominator += weight
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def build_benchmark_category_summary(rows, methods, dataset_weight_map=None):
    annotated_rows = annotate_benchmark_rows(rows)
    methods = [str(method) for method in (methods or []) if str(method).strip()]

    datasets_by_category = defaultdict(set)
    success_methods_by_dataset = defaultdict(set)
    success_datasets_by_method = defaultdict(set)
    metric_values = defaultdict(lambda: {"forecasting_rmse": [], "classification_acc": []})

    for row in annotated_rows:
        category = row.get("category") or get_dataset_category(row.get("dataset"))
        dataset = str(row.get("dataset") or "").strip()
        method = str(row.get("method") or "").strip()
        status = str(row.get("status") or "").strip().lower()

        if dataset:
            datasets_by_category[category].add(dataset)

        if status != "success" or not dataset or not method:
            continue

        success_methods_by_dataset[(category, dataset)].add(method)
        success_datasets_by_method[(category, method)].add(dataset)

        forecasting_rmse = _to_finite_float(row.get("forecasting_rmse"))
        classification_acc = _to_finite_float(row.get("classification_acc"))
        if forecasting_rmse is not None:
            metric_values[(category, method)]["forecasting_rmse"].append(
                (forecasting_rmse, _resolve_weight(row, "forecasting", dataset_weight_map))
            )
        if classification_acc is not None:
            metric_values[(category, method)]["classification_acc"].append(
                (classification_acc, _resolve_weight(row, "classification", dataset_weight_map))
            )

    categories = order_categories(datasets_by_category.keys())
    category_summary = []
    category_stats = []

    for category in categories:
        dataset_names = sorted(datasets_by_category.get(category, set()))
        all_methods_success_count = 0
        for dataset in dataset_names:
            succeeded_methods = success_methods_by_dataset.get((category, dataset), set())
            if methods and all(method in succeeded_methods for method in methods):
                all_methods_success_count += 1

        category_stats.append(
            {
                "category": category,
                "dataset_count": len(dataset_names),
                "all_methods_success_count": all_methods_success_count,
            }
        )

        for method in methods:
            method_metrics = metric_values.get((category, method), {})
            forecasting_values = method_metrics.get("forecasting_rmse", [])
            classification_values = method_metrics.get("classification_acc", [])
            category_summary.append(
                {
                    "category": category,
                    "method": method,
                    "dataset_count": len(dataset_names),
                    "success_dataset_count": len(success_datasets_by_method.get((category, method), set())),
                    "forecasting_rmse_avg": _weighted_average(forecasting_values),
                    "classification_acc_avg": _weighted_average(classification_values),
                    "aggregation": "weighted",
                }
            )

    return {
        "rows": annotated_rows,
        "categories": categories,
        "category_summary": category_summary,
        "category_stats": category_stats,
    }
