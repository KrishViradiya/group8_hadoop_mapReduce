import os
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HADOOP_DIR = "/pc2/users/v/viradiya/ds4se/week-4/hadoop/hadoop-mapreduce-project/hadoop-mapreduce-client/hadoop-mapreduce-client-core/src/main/java"

# All summaries go into one master folder to save hours of processing time
OUTPUT_DIR = "phase1_leaf_summaries"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. Gather all UNIQUE files across all 3 algorithms
# ==========================================
unique_files = set()
rsf_files = ["week3_arc.rsf", "CLEAN_acdc.rsf", "CLEAN_limbo.rsf"]

for rsf in rsf_files:
    if not os.path.exists(rsf):
        print(f"Warning: {rsf} not found. Skipping for file extraction.")
        continue
    with open(rsf, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                base_class = parts[2]
                java_path = base_class.replace('.', '/') + ".java"
                unique_files.add(java_path)

print(f"Found {len(unique_files)} unique Java files across ARC, ACDC, and LIMBO.")

# ==========================================
# 2. Load Model
# ==========================================
model_path = "ibm-granite/granite-34b-code-instruct-8k"
print(f"Loading model from: {model_path}")

hf_token = os.environ.get("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", token=hf_token)

def apply_hard_stop(text):
    stop_index = text.find("Dependencies:")
    if stop_index != -1:
        end_of_line = text.find("\n", stop_index)
        return text[:end_of_line] if end_of_line != -1 else text
    return text

# ==========================================
# 3. Process each file exactly ONCE
# ==========================================
for count, java_file_path in enumerate(list(unique_files)):
    full_java_path = os.path.join(HADOOP_DIR, java_file_path)
    safe_filename = java_file_path.replace("/", "_").replace(".java", "")
    out_file = os.path.join(OUTPUT_DIR, f"{safe_filename}.txt")
    
    if os.path.exists(out_file):
        # print(f"[{count+1}/{len(unique_files)}] Skipping (Already Done): {java_file_path}")
        continue
        
    if os.path.exists(full_java_path):
        print(f"[{count+1}/{len(unique_files)}] Zero-Shot Summarizing: {java_file_path}")
        with open(full_java_path, "r") as f:
            raw_code = f.read()
            
        tokens = tokenizer.encode(raw_code, truncation=True, max_length=512)
        safe_code = tokenizer.decode(tokens)
        
        # STRICT ZERO-SHOT PROMPT
        prompt = f"""System: You are an expert Enterprise Software Architect analyzing a MapReduce codebase.
User: Read the provided Java code. Extract a semantic summary strictly adhering to the format below. Do NOT write Java code. Do NOT add conversational text.

Required Format:
Key Functionality: [1 sentence]
Core Logic: [Brief explanation]
Inputs/Outputs: [List data]
Dependencies: [List classes]

Code:
{safe_code}

Output:"""

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        length = inputs.input_ids.shape[1] 
        outputs = model.generate(**inputs, max_new_tokens=300)
        summary = tokenizer.decode(outputs[0][length:], skip_special_tokens=True)
        
        with open(out_file, "w") as f_out:
            f_out.write(apply_hard_stop(summary).strip())
            
    else:
        print(f"[{count+1}/{len(unique_files)}] ERROR - File not found: {full_java_path}")

print("\nPhase 1 Complete! All base summaries generated for Phase 2 aggregation.")
