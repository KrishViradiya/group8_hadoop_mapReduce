import os
import csv
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 1. PATHS & DIRECTORIES
# ==========================================
LEAF_DIR = "phase1_leaf_summaries"
RSF_FILES = {
    "ARC": "week3_arc.rsf",
    "ACDC": "CLEAN_acdc.rsf",
    "LIMBO": "CLEAN_limbo.rsf"
}
OUTPUT_DIR = "final_csv_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# 3. HIERARCHY HELPERS
# ==========================================
def package_to_parts(filename: str) -> list:
    name = filename.replace(".txt", "")
    return name.split("_")

def build_tree(file_paths: list) -> dict:
    tree = {}
    for fpath in file_paths:
        basename = os.path.basename(fpath)
        parts = package_to_parts(basename)
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node.setdefault("__files__", []).append(fpath)
    return tree

# ==========================================
# 4. BOTTOM-UP HIERARCHICAL SUMMARIZER
# ==========================================
def summarize_node(node: dict, node_path: str, cluster_id: str, algo_name: str) -> str:
    combined_text = ""

    for key, child in node.items():
        if key == "__files__": continue
        child_path = f"{node_path}.{key}" if node_path else key
        child_summary = summarize_node(child, child_path, cluster_id, algo_name)
        combined_text += f"\n--- Sub-package: {child_path} ---\n{child_summary}\n"

    leaf_files = node.get("__files__", [])
    for fpath in sorted(leaf_files):
        fname = os.path.basename(fpath).replace(".txt", "")
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                combined_text += f"\n--- File: {fname} ---\n{f.read()}\n"

    if not combined_text.strip(): return ""

    tokens = tokenizer.encode(combined_text, truncation=True, max_length=4000)
    safe_text = tokenizer.decode(tokens, skip_special_tokens=True)

    # UPGRADED PROMPT 1: Forcing functional subsystem focus
    prompt = f"""System: You are an expert Enterprise Software Architect analyzing Apache Hadoop.
User: Merge the following file and package summaries (path: "{node_path}") into ONE short summary of exactly 3-4 sentences.
Do NOT list file names. Focus entirely on the shared functional role of these components. What subsystem do they form together?
Content:
{safe_text}
Summary:"""

    print(f"    [{algo_name} - Cluster {cluster_id}] Summarizing node: {node_path if node_path else 'ROOT'}")
    return run_llm(prompt, max_new_tokens=200)

# ==========================================
# 5. FINAL REPORT (ANTI-HALLUCINATION)
# ==========================================
def generate_final_report(root_summary: str, files_list: list) -> tuple:
    tokens = tokenizer.encode(root_summary, truncation=True, max_length=3000)
    safe_summary = tokenizer.decode(tokens, skip_special_tokens=True)

    # Inject up to 25 file names directly into the prompt so the LLM knows exactly what it's looking at
    files_str = "\n".join(files_list[:25])
    if len(files_list) > 25:
        files_str += f"\n...and {len(files_list) - 25} more files."

    # UPGRADED PROMPT 2: Strict title generation and context injection
    prompt = f"""System: You are an expert Enterprise Software Architect analyzing Apache Hadoop MapReduce source code.
User: Based on the provided file list and the hierarchical summary below, generate a final architectural title and description for this specific cluster.

Strict Rules for the TITLE:
- It MUST be a specific, unique functional name representing the subsystem (e.g., "Job Configuration & Lifecycle Management", "Distributed Cache Mechanism", "Map & Reduce Task Execution").
- Do NOT use generic names like "Apache Hadoop MapReduce", "MapReduce V2 API", "org.apache package", or "Hierarchical Summary".
- Do NOT hallucinate external frameworks (e.g., Apache POI) unless explicitly in the text.

Strict Rules for the DESCRIPTION:
1. Describe the Components and Interactions.
2. Describe the Quality Attributes (e.g., scalability, fault tolerance).
3. Describe the Technology Used (e.g., MapReduce, Java, HDFS).
4. MUST be concise and strictly under 150 words.

You MUST use EXACTLY this output format:
TITLE: [Your Title Here]
DESCRIPTION: [Your Description Here]

Cluster Files:
{files_str}

Merged Summary:
{safe_summary}

Output:"""

    raw_output = run_llm(prompt, max_new_tokens=250)
    
    title = "Unknown Module"
    description = raw_output
    
    if "DESCRIPTION:" in raw_output:
        parts = raw_output.split("DESCRIPTION:")
        description = parts[1].strip()
        title_part = parts[0].replace("TITLE:", "").strip()
        if title_part:
            title = title_part

    return title, description

# ==========================================
# 6. MAIN LOOP: PROCESS RSF -> BUILD CSV
# ==========================================
def safe_cluster_sort(item):
    """Safely extracts numbers from strings like 'Cluster_0' or '0' for sorting."""
    cluster_str = item[0]
    digits = "".join([char for char in cluster_str if char.isdigit()])
    return int(digits) if digits else 0

for algo_name, rsf_path in RSF_FILES.items():
    if not os.path.exists(rsf_path):
        print(f"\nSkipping {algo_name}: {rsf_path} not found.")
        continue

    print(f"\n{'='*60}\nStarting Phase 2 for Algorithm: {algo_name}\n{'='*60}")
    
    cluster_groups = defaultdict(list)
    with open(rsf_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                cluster_id = parts[1]
                base_class = parts[2]
                safe_filename = base_class.replace('.', '_') + ".txt"
                full_path = os.path.join(LEAF_DIR, safe_filename)
                cluster_groups[cluster_id].append((base_class, full_path))

    csv_filename = os.path.join(OUTPUT_DIR, f"{algo_name}_architectural_reports.csv")
    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["cluster_ID", "files", "title", "description"])

        for cluster_id, files_data in sorted(cluster_groups.items(), key=safe_cluster_sort):
            print(f"\n  Processing Cluster {cluster_id} ({len(files_data)} files)")
            
            valid_paths = [f[1] for f in files_data if os.path.exists(f[1])]
            original_classes = [f[0] for f in files_data]
            files_string = " | ".join(original_classes)
            
            if not valid_paths:
                print("    No Phase 1 summaries found. Skipping.")
                continue

            tree = build_tree(valid_paths)
            root_summary = summarize_node(tree, node_path="", cluster_id=cluster_id, algo_name=algo_name)
            
            print("    Generating final report...")
            # We now pass the original_classes list into the generator!
            title, description = generate_final_report(root_summary, original_classes)
            
            writer.writerow([cluster_id, files_string, title, description])
            csvfile.flush()

    print(f"\nFinished {algo_name}! Saved to {csv_filename}")
    
print("\nALL ALGORITHMS COMPLETE. CSV files are ready in final_csv_outputs/")
