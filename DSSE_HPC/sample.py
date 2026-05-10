import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_NAME = "ibm-granite/granite-34b-code-instruct-8k"
LIMBO_FILE = "mapreduceclientcore-1_IL_50_clusters.rsf"  # Ensure this file is in your ds4se folder
OUTPUT_FILE = "final_architecture_report.txt"

# ==========================================
# 2. FILE PARSER: Extract Clusters from RSF
# ==========================================
def load_clusters(filepath):
    clusters = defaultdict(list)
    if not os.path.exists(filepath):
        print(f"ERROR: Could not find {filepath}. Please check your folder.")
        exit()
        
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            # Standard RSF: contain cluster_ID class_name
            if len(parts) >= 3 and parts[0] == "contain":
                cluster_id = parts[1]
                class_name = parts[2]
                clusters[cluster_id].append(class_name)
                
    print(f"Successfully loaded {len(clusters)} clusters.")
    return clusters

# ==========================================
# 3. PROMPT GENERATOR (Supports RQ2)
# ==========================================
def build_prompt(cluster_id, classes, prompt_type="advanced"):
    class_list_str = "\n".join(classes)
    
    if prompt_type == "basic":
        return f"Analyze these Hadoop classes: {class_list_str}. What is the architectural name?"
        
    elif prompt_type == "advanced":
        return f"""You are a Senior Software Architect and expert in Apache Hadoop. 
Analyze these classes from {cluster_id}:

<classes>
{class_list_str}
</classes>

Based on this cluster, provide:
1. **Architectural Subsystem Name**: A technical title.
2. **Functional Responsibility**: What it manages.
3. **Architectural Assessment**: Does this grouping make logical sense? Why?"""

# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
def main():
    # Load data
    clusters = load_clusters(LIMBO_FILE)
    
    print(f"Loading Model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16
    )

    with open(OUTPUT_FILE, "w") as out_file:
        out_file.write("=== HADOOP ARCHITECTURAL RECOVERY REPORT ===\n\n")

        # Sort clusters by name (cluster_1, cluster_2...)
        sorted_keys = sorted(clusters.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)

        for cluster_id in sorted_keys:
            classes = clusters[cluster_id]
            print(f"Processing {cluster_id} ({len(classes)} classes)...")
            
            # Using the 'advanced' prompt for the best results
            prompt_text = build_prompt(cluster_id, classes, prompt_type="advanced")
            
            messages = [
                {"role": "system", "content": "You are a helpful AI architecture assistant."},
                {"role": "user", "content": prompt_text}
            ]
            
            # Correct dict-based input handling for Granite
            inputs = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt",
                return_dict=True
            ).to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Extract only the AI's new text
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Save to file
            out_file.write(f"--- {cluster_id} ---\n")
            out_file.write(f"Classes: {len(classes)}\n")
            out_file.write(f"Response:\n{response.strip()}\n\n")
            out_file.write("="*50 + "\n\n")
            
            print(f"Completed {cluster_id}")

    print(f"\nSuccess! All clusters processed. Results saved in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
