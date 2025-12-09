import json
import numpy as np
from tqdm import tqdm

def calculate_metrics(data):
    results = []
    
    # Standard IUPred threshold
    THRESHOLD = 0.5 
    # Minimum length to be considered a "Long Disordered Region"
    LDR_MIN_LEN = 30 

    for entry in tqdm(data):
        scores = np.array(entry['disorder_prediction'])
        seq_len = len(scores)
        
        # --- 1. Global Metrics ---
        mean_score = np.mean(scores)
        
        # Boolean array of disordered residues
        is_disordered = scores > THRESHOLD
        fraction_disordered = np.sum(is_disordered) / seq_len

        # --- 2. Regional Metrics (LDRs) ---
        # Find lengths of consecutive True values in is_disordered
        # We handle the edges by padding with False
        padded = np.concatenate(([False], is_disordered, [False]))
        # Find where the value changes (diff != 0)
        diffs = np.diff(padded.astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        
        segment_lengths = ends - starts
        
        if len(segment_lengths) > 0:
            longest_disordered_region = np.max(segment_lengths)
            # Count regions longer than standard domain size (e.g., 30 residues)
            num_long_regions = np.sum(segment_lengths >= LDR_MIN_LEN)
        else:
            longest_disordered_region = 0
            num_long_regions = 0

        # --- 3. Classification ---
        # A simple classification scheme
        if fraction_disordered < 0.1:
            seq_class = "Ordered"
        elif fraction_disordered > 0.9:
            seq_class = "IDP" # Fully Disordered
        elif longest_disordered_region >= 30:
            seq_class = "IDR-containing" # Contains domains
        else:
            seq_class = "Mixed"

        results.append({
            "tag": entry['tag'],
            "length": seq_len,
            "mean_disorder": round(mean_score, 4),
            "fraction_disordered": round(fraction_disordered, 4),
            "longest_disordered_region": int(longest_disordered_region),
            "num_long_regions": int(num_long_regions),
            "classification": seq_class
        })

    return results

# Example Usage with your provided snippet
# json_data = [
#  {
#    "tag": "A0A8J7XFM5.1",
#    "sequence": "MRVPVGQKIRDLELTE...",
#    "disorder_prediction": [0.635, 0.629, 0.544, 0.471, 0.446] 
#  },
#  ...
#]

if __name__ == '__main__':
    with open('aiupred_predictions.json', 'r') as f:
        json_data = json.load(f)

    results = calculate_metrics(json_data)

    with open('aiupred_stats.json', 'w') as f:
        json.dump(results, f, indent=2)

