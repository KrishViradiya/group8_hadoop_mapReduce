import argparse
import os
import subprocess
import sys


def run_jar(jar_path, file1, file2):
    if not os.path.isfile(jar_path):
        print(f"Error: Jar not found: {jar_path}")
        sys.exit(1)
    if not os.path.isfile(file1) or not os.path.isfile(file2):
        print(f"Error: One or both files do not exist: {file1}, {file2}")
        sys.exit(1)

    command = ["java", "-jar", jar_path, file1, file2]
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout.strip() if result.stdout else result.stderr.strip()
    return output


def main():
    parser = argparse.ArgumentParser(description="Compare RSF architecture files using ARCADE A2A and Cvg jars.")
    parser.add_argument("files", nargs="*", help="One or more RSF files to compare")
    parser.add_argument("--a2a-jar", default=os.path.join(os.path.dirname(__file__), "jars", "arcade_core_A2a.jar"), help="Path to arcade_core_A2a.jar")
    parser.add_argument("--cvg-jar", default=os.path.join(os.path.dirname(__file__), "jars", "arcade_core_Cvg.jar"), help="Path to arcade_core_Cvg.jar")
    args = parser.parse_args()

    if not args.files:
        args.files = [
            os.path.join("output", "ACDC", "mapreduce_ACDC.rsf"),
            os.path.join("output", "LIMBO", "mapreduceclientcore-1_IL_50_clusters.rsf"),
            os.path.join("output", "WCA", "mapreduceclientcore-1_UEMNM_100_clusters.rsf"),
        ]
        print("No input files provided. Comparing the default three files:")
        for path in args.files:
            print(f"  {path}")

    if len(args.files) < 2:
        parser.error("Provide at least two RSF files to compare.")

    print("--- STARTING JAR EVALUATION ---")
    print(f"A2A Jar: {args.a2a_jar}")
    print(f"Cvg Jar: {args.cvg_jar}")

    file_pairs = []
    for i in range(len(args.files)):
        for j in range(i + 1, len(args.files)):
            file_pairs.append((args.files[i], args.files[j]))

    for f1, f2 in file_pairs:
        print(f"\nComparing: {f1} vs {f2}\n")

        a2a_result = run_jar(args.a2a_jar, f1, f2)
        print("A2A Result:")
        print(a2a_result or "<no output>")

        cvg_result = run_jar(args.cvg_jar, f1, f2)
        print("\nCvg Result:")
        print(cvg_result or "<no output>")

    print("\n--- ALL TASKS FINISHED ---")


if __name__ == "__main__":
    main()
