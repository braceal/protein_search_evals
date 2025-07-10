import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Dict, List

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
    """Load search results from JSON, prepending best_hit to each query's hits."""
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

# Load exact and IVF results
exact_results = load_hits(exact_file)
ivf_results = {nprobe: load_hits(path) for nprobe, path in nprobe_files.items()}

# Sanity check
query_ids = set(exact_results.keys())
for res in ivf_results.values():
    assert set(res.keys()) == query_ids, "Query ID mismatch across result sets"

def compute_recall_at_k(exact: List[str], approx: List[str], k: int) -> float:
    return len(set(approx[:k]) & set(exact[:k])) / len(set(exact[:k]))

# Compute recall curves
recall_curves = defaultdict(lambda: np.zeros(len(K_VALUES)))
for nprobe, results in ivf_results.items():
    for qid in query_ids:
        exact = exact_results[qid]
        approx = results[qid]
        for i, k in enumerate(K_VALUES):
            recall_curves[nprobe][i] += compute_recall_at_k(exact, approx, k)
    recall_curves[nprobe] /= len(query_ids)

# Plot Recall@k
plt.figure(figsize=(10, 6))
for nprobe in sorted(recall_curves):
    plt.plot(K_VALUES, recall_curves[nprobe], label=f"nprobe={nprobe}")
plt.xlabel("k")
plt.ylabel("Recall@k")
plt.title("Recall@k across different nprobe values")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("recall_at_k.png", dpi=300)
plt.show()

# Plot Recall@100 vs nprobe
nprobes = sorted(recall_curves)
recall_at_100 = [recall_curves[nprobe][99] for nprobe in nprobes]

plt.figure(figsize=(8, 5))
plt.plot(nprobes, recall_at_100, marker='o')
plt.xlabel("nprobe")
plt.ylabel("Recall@100")
plt.title("Recall@100 vs nprobe")
plt.grid(True)
plt.tight_layout()
plt.savefig("recall_at_100_vs_nprobe.png", dpi=300)
plt.show()

# Plot Recall@10 vs nprobe
recall_at_10 = [recall_curves[nprobe][9] for nprobe in nprobes]  # index 9 = k=10

plt.figure(figsize=(8, 5))
plt.plot(nprobes, recall_at_10, marker='s', color='green')
plt.xlabel("nprobe")
plt.ylabel("Recall@10")
plt.title("Recall@10 vs nprobe")
plt.grid(True)
plt.tight_layout()
plt.savefig("recall_at_10_vs_nprobe.png", dpi=300)
plt.show()

# Plot Recall@5 vs nprobe
recall_at_5 = [recall_curves[nprobe][4] for nprobe in nprobes]  # index 4 = k=5

plt.figure(figsize=(8, 5))
plt.plot(nprobes, recall_at_5, marker='s', color='green')
plt.xlabel("nprobe")
plt.ylabel("Recall@5")
plt.title("Recall@5 vs nprobe")
plt.grid(True)
plt.tight_layout()
plt.savefig("recall_at_5_vs_nprobe.png", dpi=300)
plt.show()


print(f"{'nprobe':>8} | {'Recall@5':>10} | {'Recall@5':>10} | {'Recall@100':>10}")
print("-" * 33)
for nprobe in nprobes:
    r5 = recall_curves[nprobe][4]
    r10 = recall_curves[nprobe][9]
    r100 = recall_curves[nprobe][99]
    print(f"{nprobe:>8} | {r5:10.4f} | {r10:10.4f} | {r100:10.4f}")


