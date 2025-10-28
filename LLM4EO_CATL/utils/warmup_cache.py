import os, json


def _serialize_population_simple(eoh_population):
    """[{code:str, score:float}, ...]"""
    pop = getattr(eoh_population, "population", None) or []
    arr = []
    for ind in pop:
        try:
            code = str(ind)
        except Exception:
            code = ""
        try:
            sc = float(getattr(ind, "score", float("-inf")))
        except Exception:
            sc = float("-inf")
        arr.append({"code": code, "score": sc})
    return arr


def _simple_population(eoh_population):
    """[{code:str, score:float}, ...]"""
    pop = eoh_population
    arr = []
    for ind in pop:
        try:
            code = str(ind)
        except Exception:
            code = ""
        try:
            sc = float(getattr(ind, "score", float("-inf")))
        except Exception:
            sc = float("-inf")
        arr.append({"code": code, "score": sc})
    return arr


def warmup_cache_dir(exp_dir):
    return os.path.join("outputs", exp_dir, "storage", "warmup")


def warmup_cache_path(exp_dir, operator_name):
    return os.path.join(warmup_cache_dir(exp_dir), f"{operator_name}.json")


def save_warmup_batch(exp_dir, operator_name, individuals):
    path = warmup_cache_path(exp_dir, operator_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(individuals, f, ensure_ascii=False, indent=2)


def load_warmup_batch(exp_dir, operator_name):
    path = warmup_cache_path(exp_dir, operator_name)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
