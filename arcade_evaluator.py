import sys
from itertools import combinations

def parse_rsf(filepath):
    """Parses an RSF file and returns mappings of classes to clusters."""
    class_to_cluster = {}
    cluster_to_classes = {}
    
    try:
        with open(filepath, 'r') as file:
            for line in file:
                parts = line.strip().split()
                # RSF format: contain <cluster_name> <class_name>
                if len(parts) >= 3 and parts[0] == 'contain':
                    cluster_name = parts[1]
                    class_name = parts[2]
                    
                    class_to_cluster[class_name] = cluster_name
                    if cluster_name not in cluster_to_classes:
                        cluster_to_classes[cluster_name] = set()
                    cluster_to_classes[cluster_name].add(class_name)
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        sys.exit(1)
        
    return class_to_cluster, cluster_to_classes

def analyze_distribution(name, cluster_to_classes):
    """Analyzes the architecture for God Clusters and Garbage Clusters."""
    total_classes = sum(len(classes) for classes in cluster_to_classes.values())
    sizes = [len(classes) for classes in cluster_to_classes.values()]
    
    max_size = max(sizes) if sizes else 0
    mega_cluster_threshold = total_classes * 0.5 # 50% of the codebase
    singletons = sum(1 for size in sizes if size == 1)
    
    print(f"\n--- Distribution Profile: {name} ---")
    print(f"Total Clusters: {len(sizes)}")
    print(f"Largest Cluster Size: {max_size} classes ({round((max_size/total_classes)*100, 1)}% of system)")
    print(f"Single-Class Clusters (Garbage): {singletons}")
    
    if max_size >= mega_cluster_threshold:
        print(">> WARNING: Massive God Cluster Detected! <<")

def calculate_similarity(truth_dict, test_dict):
    """Calculates Pairwise F1-Score between the Ground Truth and Test clustering."""
    common_classes = set(truth_dict.keys()).intersection(set(test_dict.keys()))
    
    if not common_classes:
        return 0.0
        
    # Generate all possible pairs of classes
    all_pairs = list(combinations(common_classes, 2))
    
    truth_matches = 0
    test_matches = 0
    agreements = 0
    
    for class1, class2 in all_pairs:
        in_same_truth = truth_dict[class1] == truth_dict[class2]
        in_same_test = test_dict[class1] == test_dict[class2]
        
        if in_same_truth:
            truth_matches += 1
        if in_same_test:
            test_matches += 1
        if in_same_truth and in_same_test:
            agreements += 1
            
    # Precision: When test clusters them together, how often is it right?
    precision = agreements / test_matches if test_matches > 0 else 0
    # Recall: When truth clusters them together, how often did test find it?
    recall = agreements / truth_matches if truth_matches > 0 else 0
    
    # F1 Score: Harmonic mean of Precision and Recall
    if precision + recall == 0:
        return 0.0
        
    f1_score = 2 * (precision * recall) / (precision + recall)
    return round(f1_score * 100, 2)

if __name__ == "__main__":
    print("=== ARCADE Architecture Evaluator ===")
    
    # Define file paths (Change these to match your exact file names!)
    acdc_file = r"output\ACDC\mapreduce_ACDC.rsf"
    test_file = r"output\WCA\mapreduceclientcore-1_UEMNM__clusters.rsf"
    
    # 1. Parse the files
    truth_class_map, truth_cluster_map = parse_rsf(acdc_file)
    test_class_map, test_cluster_map = parse_rsf(test_file)
    
    # 2. Print distribution analytics
    analyze_distribution("ACDC (Ground Truth)", truth_cluster_map)
    analyze_distribution("Test Algorithm", test_cluster_map)
    
    # 3. Calculate and print Similarity Score
    score = calculate_similarity(truth_class_map, test_class_map)
    print(f"\n=======================================")
    print(f"Similarity Score (Pairwise F1): {score}%")
    print(f"=======================================")