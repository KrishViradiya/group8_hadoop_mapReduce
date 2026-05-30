# Hadoop MapReduce Client Core — Architecture Recovery

> A multi-week study applying clustering algorithms, semantic embeddings, and LLMs to recover and describe the software architecture of Apache Hadoop's MapReduce Client Core module.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Team & Repository](#team--repository)
- [Environment Setup](#environment-setup)
- [Week 1: Dependency Extraction & Clustering](#week-1-dependency-extraction--clustering)
- [Week 2: Cluster Evaluation & LLM Exploration](#week-2-cluster-evaluation--llm-exploration)
- [Week 3: Semantic Clustering](#week-3-semantic-clustering)
- [Week 4: LLM-Based Architectural Recovery (HPC)](#week-4-llm-based-architectural-recovery-hpc)
- [Key Results Summary](#key-results-summary)
- [Repository Structure](#repository-structure)

---

## Project Overview

This project addresses two core research questions in automated software architecture recovery:

1. **How do different clustering algorithms vary in their ability to determine the architectural components of a system?**
2. **How do different prompting techniques affect an LLM's ability to describe architectural components from source code?**

The target system is **Apache Hadoop**, specifically the `hadoop-mapreduce-client-core` component (version **3.4.1**), with focus on the `org.apache.hadoop.mapreduce` package.

The pipeline spans four weeks of incremental work:

| Week | Focus |
|------|-------|
| Week 1 | Dependency extraction + structural clustering (WCA, LIMBO, ACDC) |
| Week 2 | Cluster evaluation (A2A, Coverage) + lightweight LLM experiments |
| Week 3 | Semantic clustering with embedding models |
| Week 4 | Heavyweight LLM architectural recovery on HPC (A/B prompting test) |

---

## Team & Repository

- **Project:** Group 8 — Hadoop MapReduce Client Core
- **GitHub:** [https://github.com/KrishViradiya/group8_hadoop_mapReduce](https://github.com/KrishViradiya/group8_hadoop_mapReduce)

---

## Environment Setup

| Item | Detail |
|------|--------|
| Hadoop version | 3.4.1 |
| Java | JDK 17 |
| Tooling | ARCADE suite (JavaParser, Clusterer, ACDC, A2A, Coverage) |
| LLM experiments | Google Colab (lightweight), University HPC (heavyweight) |
| Scripting | Python 3 |

---

## Week 1: Dependency Extraction & Clustering

### Workflow

1. Obtain `hadoop-mapreduce-client-core-3.4.1.jar`
2. Run **ARCADE JavaParser** to produce a master RSF dependency file
3. Filter relations to retain only `org.apache.hadoop.mapreduce` and `org.apache.hadoop.mapred` namespaces
4. Apply **WCA**, **LIMBO**, and **ACDC** clustering algorithms on the filtered RSF

### RSF Format

Dependencies are stored line-by-line as:
```
depends Source_Package Target_Package
```

### Clustering Algorithms

| Algorithm | Approach | Notable Finding |
|-----------|----------|-----------------|
| **WCA** | Structural similarity (UEM / UEMNM) | Severe snowball effect; 90%+ of classes merged into one dominant cluster in several settings |
| **LIMBO** | Information-theoretic (IL) | More balanced decomposition; better separation of input, output, shuffle subsystems |
| **ACDC** | Pattern-based decomposition | Highest human readability; clearest subsystem boundaries; used as reference architecture |

### Conclusion

LIMBO and ACDC were more useful than WCA for this highly coupled subsystem. ACDC was designated the **reference architecture** for Week 2 evaluation.

---

## Week 2: Cluster Evaluation & LLM Exploration

### Evaluation

544 files were evaluated across multiple WCA and LIMBO configurations using **A2A** and **Coverage** metrics.

#### Best Configurations

| Algorithm | Serial | Stop | Clusters | Largest Cluster | Largest (%) | Avg. Size | Singletons |
|-----------|--------|------|----------|-----------------|-------------|-----------|------------|
| WCA-UEM | 90 | 25 | 90 | 31 classes | 5.70% | 6.04 | 66 |
| LIMBO-IL | 50 | 25 | 50 | 94 classes | 17.28% | 10.88 | 0 |

- **WCA** → fine-grained, high-purity clusters; limits snowball effect
- **LIMBO** → broader structural groupings; no singletons

#### Cross-Algorithm Comparison (vs. ACDC reference)

| Pair | A2A Score | Coverage (1→2) | Coverage (2→1) |
|------|-----------|----------------|----------------|
| ACDC vs LIMBO-IL-50-25 | 84.17 | 0.0435 | 0.0200 |
| ACDC vs WCA-UEM-90-75 | 83.40 | 0.1304 | 0.0333 |
| LIMBO-IL-50-25 vs WCA-UEM-90-75 | 84.50 | 0.0200 | 0.0111 |

**WCA** aligned more closely with ACDC (lower A2A, higher forward coverage). WCA and LIMBO produced fundamentally different decompositions.

### Generated CSVs

- `acdc_wca.csv`
- `acdc_limbo.csv`
- `jar_metrics.csv`

### Lightweight LLM Exploration

**Model:** `ibm-granite/granite-3.3-8b-instruct` (tested in Google Colab)

Four prompt styles were tested on the same Java file:

| # | Technique | Description |
|---|-----------|-------------|
| 1 | **Zero-shot** | Ask the model to explain the class and its architectural role |
| 2 | **Role-based** | Model acts as a senior software architect for distributed systems |
| 3 | **Context-aware** | Cluster membership info from ARCADE is provided |
| 4 | **Few-shot** | A worked example is prepended before the actual target file |

---

## Week 3: Semantic Clustering

### Methodology

1. **Embedding Extraction** — Applied `ibm-granite/granite-embedding-english-r2` and `jinaai/jina-code-embeddings-0.5b` to each source file
2. **Semantic Similarity Matrix** — Cosine similarity between embedding vectors (range: 0–1)
3. **Structural Similarity Matrix** — Built from the filtered RSF file (normalized to 0–1)
4. **Combined Distance Matrix** — Merged with weighting parameter **α**:
   ```
   combined = α × structural + (1 − α) × semantic
   distance = 1 − combined
   ```
5. **Agglomerative Clustering** — Applied to the distance matrix to produce RSF outputs

### Parameters Explored

- α ∈ {0.3, 0.5, **0.7**}
- k (cluster target) ∈ {8, 12, **16**}

Best results were obtained at **α = 0.7, k = 16** for both embedding models.

### Baseline for Comparison

`mapreduceclientcore_algo-limbo_measure-il_serial-50_stop-25.rsf`

### Visualizations

- **Granite model outputs:** [`granite-embedding-english-r2/`](https://github.com/KrishViradiya/group8_hadoop_mapReduce/tree/main/Semantic_Clustering%20%26%20Evaluation/Harsh_Savani/granite-embedding-english-r2)
- **Jina model outputs:** [`jina-code-embeddings-0.5b/`](https://github.com/KrishViradiya/group8_hadoop_mapReduce/tree/main/Semantic_Clustering%20%26%20Evaluation/Harsh_Savani/jina-code-embeddings-0.5b)

---

## Week 4: LLM-Based Architectural Recovery (HPC)

### Heavyweight Model

`ibm-granite/granite-34b-code-instruct-8k` — run on University HPC cluster

### Pipeline

**Phase 1 (Leaf Nodes):** 539 `.java` files processed individually → semantic summaries (functionality, core logic, inputs/outputs, dependencies)

**Phase 2 (Branch Nodes):** File-level summaries aggregated into 12 clusters → LLM generates architectural title + high-level descriptive report per cluster

### A/B Prompting Test (Research Question 2)

Two techniques were run in parallel on identical clusters:

| | Technique A — One-Shot | Technique B — Chain-of-Thought |
|--|------------------------|-------------------------------|
| **Adherence to cluster scope** | ✅ Strict; focused on localized logic | ❌ Tended to over-generalize |
| **Title specificity** | ✅ Specific functional titles (e.g. *"Shuffle & Sort Layer"*) | ❌ Generic (e.g. *"MapReduce System"*) |
| **Hallucination** | Low | Higher; referenced HDFS/YARN outside scope |
| **Compute cost** | Lower | Significantly higher |

**Finding:** One-Shot prompting outperformed Chain-of-Thought for localized cluster description in this context.

### Engineering Challenges

| Challenge | Mitigation |
|-----------|------------|
| CUDA Out-of-Memory (8,192 token limit) | Reduced input truncation from 6,000 → 4,500 tokens to reserve GPU memory for 2,000-token output |
| Completion hallucination (model invented Java packages) | Python "Hard Stop" string-slicing filter to truncate after final required section |
| Prompt echoing (model repeated input instead of synthesizing) | Injected negative constraints + `[END OF REPORT]` anchor |

---

## Key Results Summary

| Aspect | Best Result |
|--------|-------------|
| Structural clustering (vs. ACDC) | **WCA-UEM, serial 90, stop 25** — A2A 83.40, coverage 0.1304 |
| Balanced cluster decomposition | **LIMBO-IL, serial 50, stop 25** — no singletons, avg. size 10.88 |
| Semantic clustering | **α = 0.7, k = 16** with either embedding model |
| LLM prompting technique | **One-Shot** outperformed Chain-of-Thought for architectural focus |

> **Full project report** available in the repository as a PDF.