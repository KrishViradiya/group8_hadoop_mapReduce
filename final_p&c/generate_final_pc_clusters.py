import csv
import statistics
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from arcade_evaluator import calculate_similarity, parse_rsf


STOP_THRESHOLDS = [25, 50, 75, 100]
LIMBO_SERIALS = [25, 30, 40, 50, 60, 70, 80, 90, 100, 110]
WCA_SERIALS = [25, 50, 75, 80, 90, 100, 110, 120, 130, 140, 150]


def valid_pairs(stop_thresholds: list[int], serial_thresholds: list[int]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for stop_threshold in stop_thresholds:
        for serial_threshold in serial_thresholds:
            if serial_threshold >= stop_threshold:
                pairs.append((serial_threshold, stop_threshold))
    return pairs


def summarize_rsf(rsf_path: Path) -> dict[str, float | int]:
    _, cluster_map = parse_rsf(str(rsf_path))
    sizes = [len(classes) for classes in cluster_map.values()]
    singleton_count = sum(1 for size in sizes if size == 1)
    if not sizes:
        return {
            "cluster_count": 0,
            "total_entities": 0,
            "largest_cluster_size": 0,
            "largest_cluster_pct": 0.0,
            "smallest_cluster_size": 0,
            "average_cluster_size": 0.0,
            "singleton_count": 0,
        }

    total_entities = sum(sizes)
    largest_cluster_size = max(sizes)
    return {
        "cluster_count": len(sizes),
        "total_entities": total_entities,
        "largest_cluster_size": largest_cluster_size,
        "largest_cluster_pct": round(largest_cluster_size / total_entities * 100, 2),
        "smallest_cluster_size": min(sizes),
        "average_cluster_size": round(statistics.mean(sizes), 2),
        "singleton_count": singleton_count,
    }


def run_clusterer(
    java_exe: Path,
    jar_path: Path,
    deps_path: Path,
    output_dir: Path,
    algo: str,
    measure: str,
    serial_threshold: int,
    stop_threshold: int,
    package_prefix: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    before = set(output_dir.glob("*.rsf"))
    command = [
        str(java_exe),
        "-Xmx4096m",
        "-jar",
        str(jar_path),
        f"algo={algo}",
        "language=java",
        f"deps={deps_path}",
        f"measure={measure}",
        "projname=mapreduceclientcore",
        "projversion=1",
        f"projpath={output_dir}",
        "serial=ARCHSIZE",
        f"serialthreshold={serial_threshold}",
        "stop=PRESELECTED",
        f"stopthreshold={stop_threshold}",
        f"packageprefix={package_prefix}",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Clusterer failed for {algo}/{measure} serial={serial_threshold} stop={stop_threshold}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    after = set(output_dir.glob("*.rsf"))
    new_files = sorted(after - before, key=lambda path: path.stat().st_mtime, reverse=True)
    if not new_files:
        raise FileNotFoundError(
            f"No RSF produced for {algo}/{measure} serial={serial_threshold} stop={stop_threshold}"
        )

    generated = new_files[0]
    final_name = (
        f"mapreduceclientcore_algo-{algo.lower()}_measure-{measure.lower()}_"
        f"serial-{serial_threshold}_stop-{stop_threshold}.rsf"
    )
    final_path = output_dir / final_name
    if generated != final_path:
        if final_path.exists():
            final_path.unlink()
        generated.replace(final_path)
    return final_path


def collect_rows(
    root: Path,
    output_dir: Path,
    algo: str,
    measure: str,
    pairs: list[tuple[int, int]],
    truth_map: dict[str, str],
) -> list[dict[str, str | int | float]]:
    java_exe = Path(r"C:\Program Files\Java\jdk-17\bin\java.exe")
    jar_path = root / "jars" / "arcade_core_clusterer.jar"
    deps_path = root / "output" / "rsf" / "mapreduce_full.rsf"
    package_prefix = "org.apache.hadoop.map"

    rows: list[dict[str, str | int | float]] = []
    for serial_threshold, stop_threshold in pairs:
        rsf_path = run_clusterer(
            java_exe=java_exe,
            jar_path=jar_path,
            deps_path=deps_path,
            output_dir=output_dir,
            algo=algo,
            measure=measure,
            serial_threshold=serial_threshold,
            stop_threshold=stop_threshold,
            package_prefix=package_prefix,
        )
        class_map, _ = parse_rsf(str(rsf_path))
        stats = summarize_rsf(rsf_path)
        rows.append(
            {
                "algorithm": algo,
                "measure": measure,
                "serial_threshold": serial_threshold,
                "stop_threshold": stop_threshold,
                "filename": rsf_path.name,
                "rsf_path": str(rsf_path),
                "cluster_count": stats["cluster_count"],
                "total_entities": stats["total_entities"],
                "largest_cluster_size": stats["largest_cluster_size"],
                "largest_cluster_pct": stats["largest_cluster_pct"],
                "smallest_cluster_size": stats["smallest_cluster_size"],
                "average_cluster_size": stats["average_cluster_size"],
                "singleton_count": stats["singleton_count"],
                "similarity_score": calculate_similarity(truth_map, class_map),
            }
        )
    return rows


def write_csv(csv_path: Path, rows: list[dict[str, str | int | float]]) -> None:
    fieldnames = [
        "algorithm",
        "measure",
        "serial_threshold",
        "stop_threshold",
        "filename",
        "rsf_path",
        "cluster_count",
        "total_entities",
        "largest_cluster_size",
        "largest_cluster_pct",
        "smallest_cluster_size",
        "average_cluster_size",
        "singleton_count",
        "similarity_score",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = ROOT_DIR
    final_dir = Path(__file__).resolve().parent
    acdc_path = root / "output" / "ACDC" / "mapreduce_ACDC.rsf"
    truth_map, _ = parse_rsf(str(acdc_path))

    limbo_pairs = valid_pairs(STOP_THRESHOLDS, LIMBO_SERIALS)
    wca_pairs = valid_pairs(STOP_THRESHOLDS, WCA_SERIALS)

    limbo_output = final_dir / "LIMBO"
    wca_output = final_dir / "WCA"

    limbo_rows = collect_rows(root, limbo_output, "LIMBO", "IL", limbo_pairs, truth_map)
    wca_rows: list[dict[str, str | int | float]] = []
    for measure in ["UEM", "UEMNM"]:
        wca_rows.extend(collect_rows(root, wca_output, "WCA", measure, wca_pairs, truth_map))

    limbo_rows.sort(key=lambda row: (int(row["stop_threshold"]), int(row["serial_threshold"])))
    wca_rows.sort(key=lambda row: (row["measure"], int(row["stop_threshold"]), int(row["serial_threshold"])))

    write_csv(final_dir / "limbo_final_pc_summary.csv", limbo_rows)
    write_csv(final_dir / "wca_final_pc_summary.csv", wca_rows)

    print(f"Generated {len(limbo_rows)} LIMBO RSFs")
    print(f"Generated {len(wca_rows)} WCA RSFs")
    print(f"Output folder: {final_dir}")


if __name__ == "__main__":
    main()
