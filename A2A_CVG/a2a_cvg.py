import os
import subprocess
import sys
import csv
import re


def run_jar(jar_path, file1, file2):
    if not os.path.isfile(jar_path):
        print(f"Error: Jar not found: {jar_path}")
        sys.exit(1)
    if not os.path.isfile(file1) or not os.path.isfile(file2):
        print(f"Error: One or both files do not exist:\n  {file1}\n  {file2}")
        sys.exit(1)

    command = ["java", "-jar", jar_path, file1, file2]
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout.strip() if result.stdout else result.stderr.strip()
    return output


def extract_a2a_score(output):
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    if matches:
        return float(matches[-1])
    return None


def extract_cvg_scores(output):
    matches = re.findall(r"is\s+([-+]?\d*\.\d+|\d+)", output)
    if len(matches) >= 2:
        return float(matches[0]), float(matches[1])
    return None, None


def main():
    a2a_jar = r"D:\Paderborn University\DSSEgit\group8_hadoop_mapReduce\jars\arcade_core_A2a.jar"
    cvg_jar = r"D:\Paderborn University\DSSEgit\group8_hadoop_mapReduce\jars\arcade_core_Cvg.jar"

    files = [
        r"D:\Paderborn University\DSSEgit\group8_hadoop_mapReduce\output\ACDC\mapreduce_ACDC.rsf",
        r"D:\Paderborn University\DSSEgit\group8_hadoop_mapReduce\final_p&c\LIMBO\mapreduceclientcore_algo-limbo_measure-il_serial-50_stop-25.rsf",
        r"D:\Paderborn University\DSSEgit\group8_hadoop_mapReduce\final_p&c\WCA\mapreduceclientcore_algo-wca_measure-uem_serial-90_stop-75.rsf"
    ]

    file_pairs = [
        (files[0], files[1]),
        (files[0], files[2]),
        (files[1], files[2])
    ]

    results_data = []

    for f1, f2 in file_pairs:
        print(f"\nComparing:\n{f1} \nvs \n{f2}\n")

        a2a_result = run_jar(a2a_jar, f1, f2)
        a2a_score = extract_a2a_score(a2a_result)

        cvg_result = run_jar(cvg_jar, f1, f2)
        cvg_1_to_2, cvg_2_to_1 = extract_cvg_scores(cvg_result)

        results_data.append([os.path.basename(f1), os.path.basename(f2), a2a_score, cvg_1_to_2, cvg_2_to_1])

    csv_file = r"D:\Paderborn University\DSSEgit\group8_hadoop_mapReduce\a2a_cvg_results.csv"
    print(f"\nWriting results to {csv_file}...")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["File 1", "File 2", "A2A Score", "Cvg (File1 -> File2)", "Cvg (File2 -> File1)"])
        writer.writerows(results_data)

if __name__ == "__main__":
    main()