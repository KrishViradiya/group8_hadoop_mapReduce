# ==========================================
# 0. IMPORTS
# ==========================================
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# ==========================================
# 1. LOAD JAVA FILES
# ==========================================
JAVA_DIR = Path("hadoop/hadoop-mapreduce-project/hadoop-mapreduce-client/hadoop-mapreduce-client-core/src/main/java/org/apache/hadoop/mapreduce")

# Find all .java files recursively
java_files = list(JAVA_DIR.rglob("*.java"))
print(f"Found {len(java_files)} Java files")

# Read the content of each file
file_contents = []
file_names = []

for f in java_files:
    try:
        content = f.read_text(encoding="utf-8", errors="ignore")
        file_contents.append(content)
        file_names.append(f.stem)  # just the filename without .java
    except Exception as e:
        print(f"Could not read {f.name}: {e}")

print(f"Successfully read {len(file_contents)} files")

# ==========================================
# 2. EMBEDDING MODEL
# ==========================================
model_name = "ibm-granite/granite-embedding-english-r2"
print(f"\nLoading embedding model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model loaded! Running on: {device}")

def get_embedding(text):
    # Truncate text to 512 tokens max (model limit)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling - average all token embeddings into one vector
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.squeeze().cpu().numpy()

# ==========================================
# 3. GENERATE EMBEDDINGS
# ==========================================
print(f"\nEmbedding {len(file_contents)} Java files...")
embeddings = []

for i, content in enumerate(tqdm(file_contents)):
    try:
        emb = get_embedding(content)
        embeddings.append(emb)
    except Exception as e:
        print(f"Error embedding {file_names[i]}: {e}")
        # Use zero vector as fallback
        embeddings.append(np.zeros(768))

embeddings = np.array(embeddings)
print(f"\nEmbeddings shape: {embeddings.shape}")
print("Done embedding all files!")

# ==========================================
# 4. SEMANTIC SIMILARITY MATRIX
# ==========================================
print("\nComputing cosine similarity matrix...")

semantic_matrix = cosine_similarity(embeddings)

print(f"Semantic matrix shape: {semantic_matrix.shape}")
print(f"Min value: {semantic_matrix.min():.4f}")
print(f"Max value: {semantic_matrix.max():.4f}")

# ==========================================
# 5. STRUCTURAL SIMILARITY MATRIX
# ==========================================
print("\nBuilding structural similarity matrix from RSF file...")

RSF_PATH = r"D:\Paderborn\Study\Project-dsse-1\output\rsf\mapreduce_filtered.rsf"

# Build a set of dependencies from the RSF file
dependencies = set()

with open(RSF_PATH, "r", encoding="utf-16", errors="ignore") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "depends":
            # Handle inner classes like Cluster$1 -> Cluster
            source = parts[1].split(".")[-1].split("$")[0]
            target = parts[2].split(".")[-1].split("$")[0]
            dependencies.add((source, target))

print(f"Found {len(dependencies)} dependencies in RSF file")

# Build structural matrix
n = len(file_names)
structural_matrix = np.zeros((n, n))

for i, name_i in enumerate(file_names):
    for j, name_j in enumerate(file_names):
        if (name_i, name_j) in dependencies or (name_j, name_i) in dependencies:
            structural_matrix[i][j] = 1.0

# Normalize to 0-1
max_val = structural_matrix.max()
if max_val > 0:
    structural_matrix = structural_matrix / max_val

print(f"Structural matrix shape: {structural_matrix.shape}")
print(f"Non-zero entries: {np.count_nonzero(structural_matrix)}")

# ==========================================
# 6. COMBINE MATRICES
# ==========================================
print("\nCombining semantic and structural matrices...")

# Alpha controls the balance between structural and semantic
# 0.5 means equal weight to both
alpha = 0.5

combined_matrix = (alpha * structural_matrix) + ((1 - alpha) * semantic_matrix)

print(f"Combined matrix shape: {combined_matrix.shape}")
print(f"Min value: {combined_matrix.min():.4f}")
print(f"Max value: {combined_matrix.max():.4f}")
print("Matrices combined successfully!")

# ==========================================
# 7. AGGLOMERATIVE CLUSTERING
# ==========================================
print("\nRunning Agglomerative Clustering...")

# Convert similarity matrix to distance matrix
distance_matrix = 1 - combined_matrix

# Number of clusters to find
n_clusters = 10

clustering = AgglomerativeClustering(
    n_clusters=n_clusters,
    metric="precomputed",
    linkage="average"
)

labels = clustering.fit_predict(distance_matrix)

print(f"Clustering done!")
print(f"Number of clusters: {n_clusters}")

# Show how many files are in each cluster
unique, counts = np.unique(labels, return_counts=True)
for cluster_id, count in zip(unique, counts):
    print(f"  Cluster {cluster_id}: {count} files")
    
    
# ==========================================
# 8. SAVE RESULTS AS RSF
# ==========================================
print("\nSaving clustering results as RSF file...")

output_rsf = "arc_clustering_output.rsf"

with open(output_rsf, "w") as f:
    for file_name, cluster_id in zip(file_names, labels):
        f.write(f"contain Cluster_{cluster_id} {file_name}\n")

print(f"RSF saved to: {output_rsf}")

# ==========================================
# 9. VISUALIZE - HEATMAP
# ==========================================
print("\nGenerating heatmap...")

plt.figure(figsize=(20, 16))

sns.heatmap(
    combined_matrix,
    cmap="YlOrRd",
    xticklabels=False,
    yticklabels=False,
    cbar_kws={"label": "Similarity Score"}
)

plt.title("ARC Combined Similarity Matrix (Structural + Semantic)", 
          fontsize=16, pad=20)
plt.xlabel("Java Files", fontsize=12)
plt.ylabel("Java Files", fontsize=12)

plt.tight_layout()
plt.savefig("arc_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

print("Heatmap saved as: arc_heatmap.png")
print("\n✅ ALL DONE!")
print(f"  → arc_clustering_output.rsf")
print(f"  → arc_heatmap.png")