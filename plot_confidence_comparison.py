import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
data_path = 'test_results.json'
with open(data_path, 'r', encoding='utf-8') as f:
    results = json.load(f)

# Extract confidence scores
def get_scores(section):
    scores = results.get(section, {}).get('confidence_scores', {})
    return scores

original_scores = get_scores('original_classification')
enhanced_scores = get_scores('new_classification')

# Get all document types present in either result
doc_types = sorted(set(original_scores.keys()) | set(enhanced_scores.keys()))

orig_vals = [original_scores.get(dt, 0) for dt in doc_types]
enhanced_vals = [enhanced_scores.get(dt, 0) for dt in doc_types]

x = np.arange(len(doc_types))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, orig_vals, width, label='Original')
rects2 = ax.bar(x + width/2, enhanced_vals, width, label='Enhanced')

ax.set_ylabel('Confidence Score')
ax.set_xlabel('Document Type')
ax.set_title('Confidence Score Comparison: Original vs Enhanced Model')
ax.set_xticks(x)
ax.set_xticklabels(doc_types, rotation=45, ha='right')
ax.legend()
fig.tight_layout()

plt.savefig('confidence_comparison_enhanced.png')
plt.close()
print('Saved: confidence_comparison_enhanced.png')
