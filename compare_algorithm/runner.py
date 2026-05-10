import os
import csv
from compare_algorithm.arcade_evaluator import parse_rsf, calculate_similarity

def get_largest_cluster_info(cluster_to_classes):
    largest_name = None
    largest_size = 0
    total_classes = sum(len(classes) for classes in cluster_to_classes.values())
    
    for cluster_name, classes in cluster_to_classes.items():
        size = len(classes)
        if size > largest_size:
            largest_size = size
            largest_name = cluster_name
            
    perc = round((largest_size / total_classes * 100), 2) if total_classes > 0 else 0
    return largest_name, largest_size, perc, total_classes

def process_files(algo_name, directory, acdc_class_map):
    data = []
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return data
        
    for f in os.listdir(directory):
        if not f.endswith(".rsf"):
            continue
            
        filepath = os.path.join(directory, f)
        test_class_map, test_cluster_map = parse_rsf(filepath)
        
        # Parse filename to extract Measure and Target_Clusters
        # e.g. mapreduceclientcore-1_UEMNM_25_clusters.rsf
        parts = f.replace('.rsf', '').split('_')
        measure = parts[1] if len(parts) > 1 else ""
        target_clusters = parts[2] if len(parts) > 2 else ""
        
        actual_clusters = len(test_cluster_map)
        largest_name, largest_size, largest_perc, total_files = get_largest_cluster_info(test_cluster_map)
        
        similarity = calculate_similarity(acdc_class_map, test_class_map)
        
        data.append({
            "Algorithm": algo_name,
            "Measure": measure,
            "Target_Clusters": target_clusters,
            "Filename": f,
            "Actual_Clusters": actual_clusters,
            "Total_Files": total_files,
            "Largest_Cluster_Name": largest_name,
            "Largest_Cluster_Size": largest_size,
            "Largest_Cluster_Percentage": largest_perc,
            "Similarity_Score": similarity
        })
    
    # Sort data by Target_Clusters numerically
    data.sort(key=lambda x: int(x["Target_Clusters"]) if x["Target_Clusters"].isdigit() else 0)
    return data

def main():
    acdc_file = os.path.join("output", "ACDC", "mapreduce_ACDC.rsf")
    if not os.path.exists(acdc_file):
        print(f"Error: ACDC file not found at {acdc_file}")
        return

    # Parse ACDC ground truth
    acdc_class_map, _ = parse_rsf(acdc_file)

    # Process LIMBO and WCA directories
    limbo_data = process_files("LIMBO", os.path.join("output", "LIMBO"), acdc_class_map)
    wca_data = process_files("WCA", os.path.join("output", "WCA"), acdc_class_map)

    fieldnames = [
        "Algorithm", "Measure", "Target_Clusters", "Filename", 
        "Actual_Clusters", "Total_Files", "Largest_Cluster_Name", 
        "Largest_Cluster_Size", "Largest_Cluster_Percentage", "Similarity_Score"
    ]

    # Write CSV for LIMBO
    if limbo_data:
        with open("C:\\Users\\KEVAL\\OneDrive\\Desktop\\group8_hadoop_mapReduce\\csv\\acdc_limbo.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(limbo_data)
        print("Generated acdc_limbo.csv")

    # Write CSV for WCA
    if wca_data:
        with open("C:\\Users\\KEVAL\\OneDrive\\Desktop\\group8_hadoop_mapReduce\\csv\\acdc_wca.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(wca_data)
        print("Generated acdc_wca.csv")

if __name__ == "__main__":
    main()