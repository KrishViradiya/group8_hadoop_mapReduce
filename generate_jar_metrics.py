import os
import csv
import re
from compare import run_jar

def extract_score(output):
    """Extracts the last floating-point number from the JAR string output."""
    try:
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if matches:
            return float(matches[-1])
    except Exception:
        pass
    return 0.0

def parse_filename(filename, dir_path=""):
    """Extracts algorithm, measure, and cluster count from the RSF filename."""
    algo = "Unknown"
    if "WCA" in dir_path or "WCA" in filename: 
        algo = "WCA"
    elif "LIMBO" in dir_path or "LIMBO" in filename: 
        algo = "LIMBO"
        
    measure_match = re.search(r"_(UEMNM|UEM|IL)_", filename)
    measure = measure_match.group(1) if measure_match else "Unknown"
    
    cluster_match = re.search(r"_(\d+)_clusters", filename)
    cluster_count = cluster_match.group(1) if cluster_match else "Unknown"
    
    return algo, measure, cluster_count

def process_single_target_dir(base_file, target_dir, a2a_jar, cvg_jar, run_id_start=1):
    """Evaluates a single base file (like ACDC ground truth) against a folder of algorithms."""
    data = []
    if not os.path.exists(target_dir) or not os.path.exists(base_file):
        return data, run_id_start

    for tf in os.listdir(target_dir):
        if not tf.endswith(".rsf"):
            continue
            
        tf_path = os.path.join(target_dir, tf)
        algo, measure, cluster_count = parse_filename(tf)
        
        if algo == "Unknown":
            algo = os.path.basename(target_dir)
            
        a2a_out = run_jar(a2a_jar, base_file, tf_path)
        cvg_out = run_jar(cvg_jar, base_file, tf_path)

        data.append({
            "run_id": run_id_start,
            "algorithm": algo,
            "measure": measure,
            "stop_threshold": "",
            "serial_threshold": "",
            "cluster_count": cluster_count,
            "a2a_to_acdc": extract_score(a2a_out),
            "cvg_to_acdc": extract_score(cvg_out),
            "output_rsf": tf
        })
        run_id_start += 1
        
    return data, run_id_start

def process_cross_target_dirs(dir1, dir2, a2a_jar, cvg_jar, run_id_start=1):
    """Evaluates two folders of algorithm files against each other (e.g., WCA vs LIMBO)."""
    data = []
    if not os.path.exists(dir1) or not os.path.exists(dir2):
        return data, run_id_start
        
    files1 = [f for f in os.listdir(dir1) if f.endswith(".rsf")]
    files2 = [f for f in os.listdir(dir2) if f.endswith(".rsf")]
    
    for f1 in files1:
        f1_path = os.path.join(dir1, f1)
        algo1, measure1, cc1 = parse_filename(f1, dir1)
        
        for f2 in files2:
            f2_path = os.path.join(dir2, f2)
            algo2, measure2, cc2 = parse_filename(f2, dir2)
            
            a2a_out = run_jar(a2a_jar, f1_path, f2_path)
            cvg_out = run_jar(cvg_jar, f1_path, f2_path)
            
            data.append({
                "run_id": run_id_start,
                "algorithm": f"{algo1}_vs_{algo2}",
                "measure": f"{measure1}_vs_{measure2}",
                "stop_threshold": "",
                "serial_threshold": "",
                "cluster_count": f"{cc1}_vs_{cc2}",
                "a2a_to_acdc": extract_score(a2a_out),
                "cvg_to_acdc": extract_score(cvg_out),
                "output_rsf": f"{f1} | {f2}"
            })
            run_id_start += 1
            
    return data, run_id_start

def main():
    a2a_jar = os.path.join("jars", "arcade_core_A2a.jar")
    cvg_jar = os.path.join("jars", "arcade_core_Cvg.jar")
    acdc_file = os.path.join("output", "ACDC", "mapreduce_ACDC.rsf")
    wca_dir = os.path.join("output", "WCA")
    limbo_dir = os.path.join("output", "LIMBO")

    fieldnames = [
        "run_id", "algorithm", "measure", "stop_threshold", 
        "serial_threshold", "cluster_count", "a2a_to_acdc", 
        "cvg_to_acdc", "output_rsf"
    ]

    print("Evaluating ACDC vs WCA...")
    wca_data, current_id = process_single_target_dir(acdc_file, wca_dir, a2a_jar, cvg_jar, 1)
    
    print("Evaluating ACDC vs LIMBO...")
    limbo_data, current_id = process_single_target_dir(acdc_file, limbo_dir, a2a_jar, cvg_jar, current_id)

    print("Evaluating WCA vs LIMBO... (This might take a minute due to cross-combinations)")
    wca_limbo_data, _ = process_cross_target_dirs(wca_dir, limbo_dir, a2a_jar, cvg_jar, current_id)

    all_data = wca_data + limbo_data + wca_limbo_data

    if all_data:
        with open("jar_metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_data)
        print("Generated complete jar_metrics.csv file successfully.")

if __name__ == "__main__":
    main()