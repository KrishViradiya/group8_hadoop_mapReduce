

import os
import csv
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 1. PATHS & CONFIGURATION
# ==========================================
# Path to the actual raw Java source files (Update this to your actual hadoop directory)
SOURCE_CODE_DIR = "/pc2/users/v/viradiya/ds4se/week-4/hadoop/hadoop-mapreduce-project/hadoop-mapreduce-client/hadoop-mapreduce-client-core/src/main/java"
OUTPUT_DIR = "week4_final_csv_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Raw RSF files (The script will clean ACDC and LIMBO automatically)
RSF_FILES = {
    "ARC": "week3_arc.rsf",
    "ACDC": "acdc.rsf",   # Provide the raw one, script cleans it
    "LIMBO": "limbo.rsf"  # Provide the raw one, script cleans it
}

# Global cache to save supercomputer time.
# Even though we process cluster-by-cluster (as requested in the PDF),
# if we've already summarized a file in ARC, we reuse it for LIMBO.
LEAF_SUMMARY_CACHE = {}

# ==========================================
# 2. MODEL LOADING
# ==========================================
model_path = "ibm-granite/granite-34b-code-instruct-8k"
print(f"Loading model from: {model_path}")
hf_token = os.environ.get("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", token=hf_token)

def run_llm(prompt: str, max_new_tokens: int = 250) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    result = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    for marker in ["[END OF REPORT]", "User:", "System:"]:
        idx = result.find(marker)
        if idx != -1:
            result = result[:idx]
    return result.strip()

# ==========================================
# 3. PRE-PROCESSING (Inner Class Cleaning)
# ==========================================
def parse_and_clean_rsf(filepath: str, algo_name: str) -> dict:
    """
    Reads the RSF. If ACDC/LIMBO, strips inner classes (e.g. Job$1 -> Job)
    Returns: { cluster_id: set_of_file_paths }
    """
    cluster_dict = defaultdict(set)
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return cluster_dict

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                cluster_id = parts[1]
                class_path = parts[2]

                # CRITICAL PRE-PROCESSING: Remove inner class '$' delimiters
                if algo_name in ["ACDC", "LIMBO"]:
                    class_path = class_path.split('$')[0]

                cluster_dict[cluster_id].add(class_path)
    return cluster_dict

# ==========================================
# 4. HIERARCHY TREE BUILDER
# ==========================================
def build_tree(class_list: set) -> dict:
    """Converts a flat list of Java package paths into a nested dictionary tree."""
    tree = {}
    for class_path in class_list:
        parts = class_path.split('.')
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node.setdefault("__files__", []).append(class_path)
    return tree

# ==========================================
# 5. BOTTOM-UP PROCESSING (Following the PDF flow)
# ==========================================
def get_java_source(class_path: str) -> str:
    """Simulates reading the raw .java file from disk based on package path."""
    # Convert org.apache... to org/apache/... .java
    file_path = os.path.join(SOURCE_CODE_DIR, class_path.replace('.', '/') + ".java")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return "// Source code not found for " + class_path

def process_node(node: dict, node_path: str, cluster_id: str, algo_name: str) -> str:
    """
    Recursively processes branch nodes bottom-up.
    1. Processes Leaf nodes (raw java -> semantic summary)
    2. Processes Branch nodes (merges child summaries)
    """
    combined_summaries = ""

    # STEP 1: Process child sub-directories (Bottom-Up Recursion)
    for key, child in node.items():
        if key == "__files__": continue
        child_path = f"{node_path}.{key}" if node_path else key
        child_summary = process_node(child, child_path, cluster_id, algo_name)
        combined_summaries += f"\n--- Sub-directory: {child_path} ---\n{child_summary}\n"

    # STEP 2: Process Leaf Nodes (Files) specifically for THIS cluster (As per PDF)
    leaf_files = node.get("__files__", [])
    for class_path in sorted(leaf_files):
        # Use cache to save HPC time, but logically we are doing it inside the cluster loop
        if class_path not in LEAF_SUMMARY_CACHE:
            print(f"      [{algo_name}] Extracting semantic summary for raw file: {class_path}")
            raw_code = get_java_source(class_path)

            # Truncate raw code to fit context
            tokens = tokenizer.encode(raw_code, truncation=True, max_length=5000)
            safe_code = tokenizer.decode(tokens, skip_special_tokens=True)

            leaf_prompt = f"""System: You are an expert Java Architect.
User: Analyze the following raw Java source code and extract a semantic summary detailing:
1. Key functionality
2. Core logic
3. Inputs/Outputs
4. Dependencies

Source Code:
{safe_code}

Semantic Summary:"""
            LEAF_SUMMARY_CACHE[class_path] = run_llm(leaf_prompt, max_new_tokens=200)

        # Add to our combined rolling summary
        combined_summaries += f"\n--- File: {class_path} ---\n{LEAF_SUMMARY_CACHE[class_path]}\n"

    if not combined_summaries.strip(): return ""

    # STEP 3: Process Branch Nodes (Directories)
    # If this is the root node (node_path is empty), we skip intermediate summary and do final formatting
    if not node_path:
        return combined_summaries

    # Intermediate Branch node summary
    tokens = tokenizer.encode(combined_summaries, truncation=True, max_length=4000)
    safe_summaries = tokenizer.decode(tokens, skip_special_tokens=True)

    branch_prompt = f"""System: You are an expert Software Architect.
User: Based on the following file and subdirectory summaries for the package "{node_path}", generate a high-level summary of exactly 3-4 sentences explaining the module's overall behavior.
Do NOT pass raw code.
Summaries:
{safe_summaries}
High-Level Summary:"""

    print(f"    [{algo_name} - Cluster {cluster_id}] Summarizing branch: {node_path}")
    return run_llm(branch_prompt, max_new_tokens=150)


# ==========================================
# 6. GENERATE FINAL REPORT (STRICT PROFESSOR CONSTRAINTS)
# ==========================================
def format_final_cluster_report(root_summary: str) -> tuple:
    """Enforces the strict requirements: Components, Quality Attributes, Technology, <150 words."""
    tokens = tokenizer.encode(root_summary, truncation=True, max_length=5000)
    safe_summary = tokenizer.decode(tokens, skip_special_tokens=True)

    prompt = f"""System: You are an expert Enterprise Software Architect.
User: Based on the hierarchical architectural summary below, generate a final title and description for this cluster.

You MUST follow these strict rules:
1. State the Components and Interactions (how parts work together).
2. State the Quality Attributes (e.g., scalability, security, fault tolerance).
3. State the Technology Used (e.g., Java, frameworks, Hadoop).
4. Conciseness: The description MUST be under 150 words.

You MUST use EXACTLY this output format:
TITLE: [Your Specific Functional Title]
DESCRIPTION: [Your Description Here]

Hierarchical Summary:
{safe_summary}

Output:"""

    raw_output = run_llm(prompt, max_new_tokens=200)

    title = "Unknown Module"
    description = raw_output

    if "DESCRIPTION:" in raw_output:
        parts = raw_output.split("DESCRIPTION:")
        description = parts[1].strip()
        title_part = parts[0].replace("TITLE:", "").strip()
        if title_part: title = title_part

    return title, description

# ==========================================
# 7. MAIN EXECUTION LOOP
# ==========================================
def safe_cluster_sort(item):
    digits = "".join([char for char in item[0] if char.isdigit()])
    return int(digits) if digits else 0

print("Starting Unified Hierarchical Pipeline...\n" + "="*50)

for algo_name, rsf_path in RSF_FILES.items():
    print(f"\nProcessing Algorithm: {algo_name} (File: {rsf_path})")

    # 1. Parse and Clean RSF (Handles ACDC & LIMBO inner classes automatically)
    cluster_groups = parse_and_clean_rsf(rsf_path, algo_name)
    if not cluster_groups: continue

    csv_filename = os.path.join(OUTPUT_DIR, f"{algo_name}_architectural_reports.csv")
    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Required Submission Format
        writer.writerow(["cluster_ID", "files", "title", "description"])

        # 2. Iterate per cluster (Aligns with PDF instructions)
        for cluster_id, class_list in sorted(cluster_groups.items(), key=safe_cluster_sort):
            print(f"\n  -> Starting Cluster {cluster_id} ({len(class_list)} files)")

            # Build Tree
            tree = build_tree(class_list)

            # Bottom-Up Processing (Includes Leaf node raw-code summarization)
            root_summaries = process_node(tree, node_path="", cluster_id=cluster_id, algo_name=algo_name)

            # Generate Final Formatted Report
            print(f"  -> Formatting Final Output for Cluster {cluster_id}...")
            title, description = format_final_cluster_report(root_summaries)

            # Join files list for CSV
            files_string = " | ".join(sorted(class_list))
            writer.writerow([cluster_id, files_string, title, description])
            csvfile.flush()

    print(f"Finished {algo_name}! Data saved to {csv_filename}")

print("\nPipeline Complete. All 3 CSV files are ready.")