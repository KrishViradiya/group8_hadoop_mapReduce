import argparse
import json
import shutil
from pathlib import Path, PurePosixPath


DEFAULT_RSF = Path(
    r"D:\Paderborn University\DSSEgit\group8_hadoop_mapReduce"
    r"\Semantic_Clustering & Evaluation\Harsh_Savani"
    r"\jina-code-embeddings-0.5b\alpha_0.7_clusters_12\output.rsf"
)

DEFAULT_JSON = Path(
    r"D:\Paderborn University\DSSEgit\group8_hadoop_mapReduce"
    r"\Semantic_Clustering & Evaluation\Dhruv_Savani"
    r"\individual_class_analysis_with_cluster_role_base.json"
)

DEFAULT_OUTPUT = Path(
    r"D:\Paderborn University\DSSEgit\group8_hadoop_mapReduce"
    r"\Semantic_Clustering & Evaluation\Dhruv_Savani"
    r"\individual_class_analysis_with_cluster_role_base_updated.json"
)


def load_rsf_clusters(rsf_path):
    full_name_to_cluster = {}
    simple_name_to_clusters = {}

    with rsf_path.open("r", encoding="utf-8") as rsf_file:
        for line_number, raw_line in enumerate(rsf_file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Invalid RSF line {line_number}: {raw_line!r}")

            _, cluster_name, full_class_name = parts[:3]
            full_name_to_cluster[full_class_name] = cluster_name

            simple_name = full_class_name.rsplit(".", 1)[-1]
            simple_name_to_clusters.setdefault(simple_name, set()).add(cluster_name)

    return full_name_to_cluster, simple_name_to_clusters


def class_name_from_java_path(java_path):
    if not java_path:
        return None

    normalized_path = str(java_path).replace("\\", "/")
    path = PurePosixPath(normalized_path)
    parts = path.parts

    java_root_index = None
    for index in range(len(parts) - 3):
        if parts[index : index + 4] == ("src", "main", "java", "org"):
            java_root_index = index + 3
            break

    if java_root_index is None and "java" in parts:
        java_root_index = len(parts) - 1 - parts[::-1].index("java")

    if java_root_index is None:
        return None

    class_parts = list(parts[java_root_index:])
    if not class_parts or not class_parts[-1].endswith(".java"):
        return None

    class_parts[-1] = class_parts[-1][:-5]
    return ".".join(class_parts)


def resolve_cluster(entry_key, entry_value, full_name_to_cluster, simple_name_to_clusters):
    full_class_name = class_name_from_java_path(entry_value.get("path"))
    if full_class_name and full_class_name in full_name_to_cluster:
        return full_name_to_cluster[full_class_name], "full_path"

    simple_name = entry_key.rsplit(".", 1)[-1]
    clusters = simple_name_to_clusters.get(simple_name, set())
    if len(clusters) == 1:
        return next(iter(clusters)), "simple_name"

    return None, "not_found" if not clusters else "ambiguous"


def update_clusters(json_path, rsf_path, output_path=None, in_place=False):
    full_name_to_cluster, simple_name_to_clusters = load_rsf_clusters(rsf_path)

    with json_path.open("r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    stats = {
        "updated": 0,
        "unchanged": 0,
        "full_path_matches": 0,
        "simple_name_matches": 0,
        "not_found": [],
        "ambiguous": [],
    }

    for entry_key, entry_value in data.items():
        if not isinstance(entry_value, dict):
            stats["unchanged"] += 1
            continue

        cluster_name, match_type = resolve_cluster(
            entry_key,
            entry_value,
            full_name_to_cluster,
            simple_name_to_clusters,
        )

        if cluster_name is None:
            stats[match_type].append(entry_key)
            stats["unchanged"] += 1
            continue

        if entry_value.get("cluster") != cluster_name:
            entry_value["cluster"] = cluster_name
            stats["updated"] += 1
        else:
            stats["unchanged"] += 1

        if match_type == "full_path":
            stats["full_path_matches"] += 1
        elif match_type == "simple_name":
            stats["simple_name_matches"] += 1

    if in_place:
        backup_path = json_path.with_suffix(json_path.suffix + ".bak")
        shutil.copy2(json_path, backup_path)
        target_path = json_path
    else:
        target_path = output_path or DEFAULT_OUTPUT
        backup_path = None

    with target_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=4, ensure_ascii=False)
        output_file.write("\n")

    return target_path, backup_path, stats


def main():
    parser = argparse.ArgumentParser(
        description="Replace JSON cluster values using class-to-cluster mappings from an RSF file."
    )
    parser.add_argument("--rsf", type=Path, default=DEFAULT_RSF)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the JSON file after creating a .bak backup.",
    )
    args = parser.parse_args()

    target_path, backup_path, stats = update_clusters(
        json_path=args.json,
        rsf_path=args.rsf,
        output_path=args.output,
        in_place=args.in_place,
    )

    print(f"Written: {target_path}")
    if backup_path:
        print(f"Backup:  {backup_path}")
    print(f"Updated: {stats['updated']}")
    print(f"Unchanged: {stats['unchanged']}")
    print(f"Full path matches: {stats['full_path_matches']}")
    print(f"Simple name matches: {stats['simple_name_matches']}")
    print(f"Not found: {len(stats['not_found'])}")
    print(f"Ambiguous: {len(stats['ambiguous'])}")

    if stats["not_found"]:
        print("\nClasses not found in RSF:")
        for class_name in stats["not_found"]:
            print(f"  - {class_name}")

    if stats["ambiguous"]:
        print("\nClasses with ambiguous simple-name matches:")
        for class_name in stats["ambiguous"]:
            print(f"  - {class_name}")


if __name__ == "__main__":
    main()
