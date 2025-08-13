import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

# Hardcoded files
exact_file = "UP000000625_83333-search-results-trembl-esm3b-faesm-ubinary-exact-bs128-topk100.json"

nprobe_files = {
    1: "UP000000625_83333-search-results-trembl-esm3b-faesm-bs128-ubinary-ivf-nprobe1-topk100.json",
    2: "UP000000625_83333-search-results-trembl-esm3b-faesm-bs128-ubinary-ivf-nprobe2-topk100.json",
    4: "UP000000625_83333-search-results-trembl-esm3b-faesm-bs128-ubinary-ivf-nprobe4-topk100.json",
    8: "UP000000625_83333-search-results-trembl-esm3b-faesm-bs128-ubinary-ivf-nprobe8-topk100.json",
    16: "UP000000625_83333-search-results-trembl-esm3b-faesm-bs128-ubinary-ivf-nprobe16-topk100.json",
    32: "UP000000625_83333-search-results-trembl-esm3b-faesm-bs128-ubinary-ivf-nprobe32-topk100.json",
    64: "UP000000625_83333-search-results-trembl-esm3b-faesm-bs128-ubinary-ivf-nprobe64-topk100.json",
    128: "UP000000625_83333-search-results-trembl-esm3b-faesm-bs128-ubinary-ivf-nprobe128-topk100.json",
    256: "UP000000625_83333-search-results-trembl-esm3b-faesm-bs128-ubinary-ivf-nprobe256-topk100.json",
}

K_VALUES = list(range(1, 101))

def load_hits(filepath: str) -> Dict[str, List[str]]:
    """Load search results and prepend best_hit to hits list."""
    with open(filepath, "r") as f:
        data = json.load(f)

    results: Dict[str, List[str]] = {}
    for entry in data["hits"]:
        query_id: str = entry["query_id"]
        best_id: str = entry["best_hit"]["id"]
        hits: List[str] = [hit["id"] for hit in entry["hits"]]
        full_hits: List[str] = [best_id] + hits
        results[query_id] = full_hits
    return results

# Load exact top hit (ground truth) and IVF results
exact_hits = load_hits(exact_file)
ivf_hits = {nprobe: load_hits(path) for nprobe, path in nprobe_files.items()}

query_ids = set(exact_hits.keys())
for hits in ivf_hits.values():
    assert set(hits.keys()) == query_ids, "Query mismatch"

def reciprocal_rank_at_k(true_id: str, hit_ids: List[str], k: int) -> float:
    try:
        rank = hit_ids[:k].index(true_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0

# Compute MRR@k
mrr_curves = defaultdict(lambda: np.zeros(len(K_VALUES)))
for nprobe, results in ivf_hits.items():
    for qid in query_ids:
        true_id = exact_hits[qid][0]  # best hit from exact
        approx = results[qid]
        for i, k in enumerate(K_VALUES):
            mrr_curves[nprobe][i] += reciprocal_rank_at_k(true_id, approx, k)
    mrr_curves[nprobe] /= len(query_ids)

# Plot MRR@k
plt.figure(figsize=(10, 6))
for nprobe in sorted(mrr_curves):
    plt.plot(K_VALUES, mrr_curves[nprobe], label=f"nprobe={nprobe}")
plt.xlabel("k")
plt.ylabel("MRR@k")
plt.title("Mean Reciprocal Rank@k across different nprobe values")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mrr_at_k.png", dpi=300)
plt.show()

# Plot MRR@100 vs nprobe
nprobes = sorted(mrr_curves)
mrr_at_100 = [mrr_curves[nprobe][99] for nprobe in nprobes]

plt.figure(figsize=(8, 5))
plt.plot(nprobes, mrr_at_100, marker='o')
plt.xlabel("nprobe")
plt.ylabel("MRR@100")
plt.title("MRR@100 vs nprobe")
plt.grid(True)
plt.tight_layout()
plt.savefig("mrr_at_100_vs_nprobe.png", dpi=300)
plt.show()

# Summary print
print(f"{'nprobe':>8} | {'MRR@100':>10}")
print("-" * 22)
for nprobe in nprobes:
    print(f"{nprobe:>8} | {mrr_curves[nprobe][99]:10.4f}")

