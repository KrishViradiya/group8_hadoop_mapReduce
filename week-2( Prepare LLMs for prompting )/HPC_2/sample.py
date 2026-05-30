
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 0. READ AND GROUP ALL CLUSTERS
# ==========================================
rsf_file = "mapreduceclientcore-1_UEMNM_100_clusters.rsf"
clusters = {}

try:
    with open(rsf_file, 'r') as f:
        for line in f:
            if line.startswith("contain "):
                parts = line.strip().split()
                if len(parts) >= 3:
                    cluster_id = parts[1]
                    class_name = parts[2]
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].append(class_name)
    print(f"Successfully loaded {len(clusters)} clusters from RSF file.")
except FileNotFoundError:
    print(f"Error: Could not find {rsf_file}. Make sure it is in the same directory.")
    sys.exit(1)

# ==========================================
# 1. ENVIRONMENT & MODEL SETUP
# ==========================================
model_name = "ibm-granite/granite-34b-code-instruct-8k"

hf_token = os.environ.get('HF_TOKEN')
if not hf_token:
    print("WARNING: HF_TOKEN not found. Proceeding with open-weights model download.")

# ==========================================
# 2. LOAD THE TOKENIZER & MODEL (ONLY ONCE)
# ==========================================
print(f"Loading Tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print("Loading Model across 2x A100 GPUs...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# ==========================================
# 3. SEQUENTIAL INFERENCE LOOP
# ==========================================
# Loop through every cluster we found in the RSF file
for cluster_id, classes_in_cluster in clusters.items():

    # Format the classes cleanly
    class_list_str = "\n".join(classes_in_cluster)
    print(f"\n" + "="*50)
    print(f"Starting analysis for Cluster {cluster_id} ({len(classes_in_cluster)} classes)...")

    # Design the prompt for this specific cluster
    messages = [
        {
            "role": "system",
            "content": "You are an expert software architect specializing in the Apache Hadoop ecosystem. Your task is to analyze Java source code from the Hadoop MapReduce 'Client core' component and identify its architectural purpose."
        },
        {
            "role": "user",
            "content": f"""Please analyze the following classes from Cluster {cluster_id}:

### Task:
1. Provide a concise **Architectural Title** for this cluster.
2. Write a **High-Level Descriptive Summary** (3-4 sentences) explaining how these classes interact to support Hadoop MapReduce Client core functionality.

### Classes in Cluster {cluster_id}:
{class_list_str}"""
        }
    ]

    # Convert the prompt into tokens
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # Model inference
    print("Generating response...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.5,
        top_p=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and print to log
    input_length = inputs['input_ids'].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    # Save the output to a unique text file for this cluster
    output_filename = f"summary_cluster_{cluster_id}.txt"
    with open(output_filename, "w") as out_file:
        out_file.write(f"Cluster {cluster_id} Architectural Summary\n")
        out_file.write("="*50 + "\n\n")
        out_file.write(response)

    print(f"Successfully saved output to {output_filename}")

print("\nAll clusters processed successfully!")






