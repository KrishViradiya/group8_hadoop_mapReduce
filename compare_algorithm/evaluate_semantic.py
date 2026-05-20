import os
import csv
import sys

# Add the project root directory to sys.path to resolve the ModuleNotFoundError
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare_algorithm.compare import run_jar, extract_a2a_score, extract_cvg_scores

def find_rsf_files(base_dir):
    rsf_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".rsf"):
                rsf_files.append(os.path.join(root, file))
    return rsf_files

def main():
    baseline_file = r"D:\Upb\dsse\group8_hadoop_mapReduce\output\WCA\mapreduceclientcore_algo-wca_measure-uem_serial-90_stop-75.rsf"
    target_dir = r"D:\Upb\dsse\group8_hadoop_mapReduce\Semantic_Clustering & Evaluation\Harsh_Savani"
    a2a_jar = r"D:\Upb\dsse\group8_hadoop_mapReduce\jars\arcade_core_A2a.jar"
    cvg_jar = r"D:\Upb\dsse\group8_hadoop_mapReduce\jars\arcade_core_Cvg.jar"
    
    output_csv = os.path.join(target_dir, "semantic_comparison_results.csv")
    
    rsf_files = find_rsf_files(target_dir)
    
    if not rsf_files:
        print(f"No .rsf files found in {target_dir}")
        return
        
    results_data = []
    
    print(f"Found {len(rsf_files)} .rsf files. Starting evaluation...")
    print(f"Baseline: {os.path.basename(baseline_file)}")
    
    for i, target_file in enumerate(rsf_files):
        print(f"\n[{i+1}/{len(rsf_files)}] Comparing against: {os.path.basename(target_file)}")
        
        a2a_result = run_jar(a2a_jar, baseline_file, target_file)
        a2a_score = extract_a2a_score(a2a_result)
        print(f"  A2A Score: {a2a_score}")
        
        cvg_result = run_jar(cvg_jar, baseline_file, target_file)
        cvg_1_to_2, cvg_2_to_1 = extract_cvg_scores(cvg_result)
        print(f"  Cvg (Baseline -> Target): {cvg_1_to_2}")
        print(f"  Cvg (Target -> Baseline): {cvg_2_to_1}")
        
        results_data.append({
            "Baseline File": os.path.basename(baseline_file),
            "Target File": os.path.basename(target_file),
            "Target Path": target_file,
            "A2A Score": a2a_score,
            "Cvg (Baseline -> Target)": cvg_1_to_2,
            "Cvg (Target -> Baseline)": cvg_2_to_1
        })
        
    print(f"\nWriting results to {output_csv}...")
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Baseline File", "Target File", "Target Path", "A2A Score", "Cvg (Baseline -> Target)", "Cvg (Target -> Baseline)"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)
        
    print("\n--- ALL TASKS FINISHED ---")

if __name__ == "__main__":
    main()