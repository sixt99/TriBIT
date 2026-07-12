import subprocess
import argparse
import json
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('--exe_path', type=str, required = True)
parser.add_argument('--data_path', type=str, required = True)
parser.add_argument('--denyfile_path', type=str)
parser.add_argument('--partition_file', type=str, required = True)
parser.add_argument('--out_path', type=str, default = 'results.csv')
parser.add_argument('--n_repetitions', type=int, default = 1)
parser.add_argument('--dry_run', action='store_true')
args = parser.parse_args()

def read_partition_json(path):
    partitions_dict = {}
    if args.partition_file:
        with open(path) as f:
            partitions_dict = json.load(f)
    return partitions_dict

def output_parser_tribit(output: str) -> str:
    fields = {
        "graph": r"Graph\s*:\s*(\S+)",
        "nodes": r"Nodes \(declared\)\s*:\s*(\d+)",
        "arcs": r"Arcs \(declared\)\s*:\s*(\d+)",
        "graph_memory_gib": r"Graph memory \(GiB\)\s*:\s*([\d.]+)",
        "load_time": r"Load time \(s\)\s*:\s*([\d.]+)",
        "partition": r"Partition\s*:\s*(\d+)",
        "tasks_processed": r"Tasks processed\s*:\s*(\d+)",
        "triangles": r"Triangles\s*:\s*(\d+)",
        "counting_time": r"Counting time \(s\)\s*:\s*([\d.]+)",
        "gpus": r"GPUs\s*:\s*(\d+)",
        "triangles_per_sec": r"Triangles/sec\s*:\s*([\d.eE+-]+)",
    }
    values = {}
    for key, pattern in fields.items():
        match = re.search(pattern, output)
        if not match:
            raise ValueError(f"Could not find '{key}' in output:\n{output}")
        values[key] = match.group(1)

    return ",".join(
        values[k]
        for k in (
            "graph",
            "nodes",
            "arcs",
            "graph_memory_gib",
            "load_time",
            "partition",
            "tasks_processed",
            "triangles",
            "counting_time",
            "gpus",
            "triangles_per_sec",
        )
    )

header = "graph,nodes,arcs,graph_memory_gib,load_time,partition,tasks_processed,triangles,counting_time,gpus,triangles_per_sec"
make_execution_args = lambda full_path, partition: ["srun", args.exe_path, full_path, str(partition)]
output_parser = output_parser_tribit

partition_dict = read_partition_json(args.partition_file)

if args.denyfile_path:
    with open(args.denyfile_path, 'r') as file:
        denylist = [x.strip() for x in file.readlines()]

counter = 0
with open(args.out_path, 'a+') as file:
    file.seek(0)
    first_line = file.readline().strip()
    if first_line != header:
        file.write(header + "\n")
        file.flush()

    # Walk over all possible files ending with .graph 
    for root, dirs, files in os.walk(args.data_path):
        for name in files:
            full_path = os.path.join(root, name)
            if full_path.endswith(".graph"):
                # Check if .properties exist in the same folder
                if not os.path.exists(full_path[:-len(".graph")] + ".properties"):
                    continue

                # Skip blacklisted matrices
                if args.denyfile_path:
                    if any(re.search(pattern, name) for pattern in denylist):
                        continue
                
                if (args.dry_run):
                    print(counter, name, flush = True)
                    counter += 1
                    continue
                
                try:
                    # Run execution 
                    for rep in range(args.n_repetitions):
                        # Read partition assigned to this graph
                        graph_key = name.split(".")[0]
                        partition = partition_dict.get(graph_key)
                        if partition is None:
                            print(f"[WARNING] No partition found for '{graph_key}', defaulting to 64")
                            partition = 64
                        
                        # Launch triangle execution
                        result = subprocess.run(
                            make_execution_args(os.path.splitext(full_path)[0], partition),
                            capture_output=True,
                            text=True,
                            check=True # Raise an exception if the program fails
                        )
                        result = output_parser(result.stdout)
                        print(counter, name, flush = True)
                        counter += 1

                        # Write results (only when there is no exception)
                        file.write(f"{result}\n")
                        file.flush()

                except Exception as e:
                    print(f"Execution failed for {full_path}", flush = True)
                    print(f"* exception type: {type(e).__name__}", flush = True)
                    print(f"* exception message: {e}", flush = True)
                    if isinstance(e, subprocess.CalledProcessError):
                        print("* stdout:", e.stdout, flush = True)
                        print("* stderr:", e.stderr, flush = True)
                        print("* returncode:", e.returncode, flush = True)

print("Total amount of matrices:", counter)
