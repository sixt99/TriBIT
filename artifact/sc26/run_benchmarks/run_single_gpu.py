import pandas as pd
import subprocess
import argparse
import tempfile
import sqlite3
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('--exe_path', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--denyfile_path', type=str)
parser.add_argument('--out_path', type=str, default='results.csv')
parser.add_argument('--n_repetitions', type=int, default=1)
parser.add_argument('--dry_run', action='store_true')
parser.add_argument('--get_memory_consumption', action='store_true')
parser.add_argument('--max_matrices', type=int, required=False)
parser.add_argument(
    '--preprocess',
    action='store_true',
    help=(
        "Rewrite each .mtx file in place before running it: convert "
        "real/integer/complex MatrixMarket headers to 'pattern' and drop "
        "everything past the first two columns of each entry. This "
        "PERMANENTLY MODIFIES the input files on disk, so it is opt-in."
    ),
)
args = parser.parse_args()

print("Exe: ", args.exe_path)


# Get 8 random bytes from the OS entropy pool and convert to an integer
def random_integer():
    return int.from_bytes(os.urandom(8), "big")


def mtx_file_is_valid(path):
    is_valid = False
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments
            if not line or line.startswith("%"):
                continue
            # First non-comment line is: rows cols nnz
            line = list(map(int, line.split()))
            # If the number is not good, just return
            if len(line) != 3:
                return False, -1, -1, -1
            nrows, ncols, nnz = line
            # The matrix should be square!
            if nrows == ncols:
                is_valid = True
            return is_valid, nrows, ncols, nnz
    return is_valid, -1, -1, -1


def keep_first_two_columns(input_path):
    """Rewrite an .mtx file in place: convert its value type to 'pattern'
    and keep only the first two (row, col) fields of every entry line.
    Only called when --preprocess is passed, since it mutates the file."""
    first = True
    with tempfile.NamedTemporaryFile("w", delete=False, dir=".") as tmp:
        with open(input_path, "r") as infile:
            for line in infile:
                if line.startswith("%"):
                    if line.startswith("%%MatrixMarket"):
                        if "real" not in line and "integer" not in line and "complex" not in line:
                            os.remove(tmp.name)
                            return
                        new_header = line.replace("real", "pattern")
                        new_header = new_header.replace("integer", "pattern")
                        new_header = new_header.replace("complex", "pattern")
                        tmp.write(new_header)
                    else:
                        tmp.write(line)
                else:
                    if first:
                        first = False
                        tmp.write(line)  # First line is nnz/shape, keep it
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        new_line = f"{parts[0]} {parts[1]}\n"
                    else:
                        new_line = line  # fallback safety
                    tmp.write(new_line)
    os.replace(tmp.name, input_path)


def get_sqlite3_connection(full_path, report_name, make_execution_args):
    nsys_launch_args = [
        "nsys", "profile",
        "--trace", "cuda",
        "--cuda-memory-usage", "true",
        "--export", "sqlite",
        "-f", "true",
        "-o", report_name
    ] + make_execution_args(full_path)
    subprocess.run(
        nsys_launch_args,
        capture_output=False,
        text=False,
        check=True  # raises an exception if the program fails
    )
    os.remove(report_name + ".nsys-rep")
    conn = sqlite3.connect(report_name + ".sqlite")
    return conn


def get_max_memory_consumption(full_path, report_id, make_execution_args):
    """Cumulative-allocation based peak memory tracking."""
    report_name = f"report_{report_id}"
    conn = get_sqlite3_connection(full_path, report_name, make_execution_args)
    df = pd.read_sql(
        '''
            SELECT bytes, memoryOperationType
            FROM CUDA_GPU_MEMORY_USAGE_EVENTS
            WHERE memKind = 2
            ORDER BY start;
        ''', conn)
    conn.close()
    os.remove(report_name + ".sqlite")
    df['signed_bytes'] = df['bytes'].where(df['memoryOperationType'] == 0, -df['bytes'])
    cumulative = df['signed_bytes'].cumsum()
    peak = cumulative.max()
    return peak


def get_max_memory_consumption_pool(full_path, report_id, make_execution_args):
    """Memory-pool based peak memory tracking."""
    report_name = f"report_{report_id}"
    conn = get_sqlite3_connection(full_path, report_name, make_execution_args)
    df = pd.read_sql(
        '''
            SELECT localMemoryPoolUtilizedSize
            FROM CUDA_GPU_MEMORY_USAGE_EVENTS;
        ''', conn)
    conn.close()
    os.remove(report_name + ".sqlite")
    peak = int(df.dropna().to_numpy().flatten()[2:].max())  # skip first two dummy allocations
    return peak


# ---------------------------------------------------------------------------
# Per-executable output parsers
# ---------------------------------------------------------------------------

def output_parser_tribit(output: str, full_path: str, exe_path: str) -> str:
    fields = {
        "n_blocks": r"Blocks of threads\s*:\s*(\d+)",
        "preprocessing_time": r"Preprocessing \(ms\)\s*:\s*([\d.]+)",
        "kernel_time": r"Kernel \(ms\)\s*:\s*([\d.]+)",
        "triangles": r"Triangles\s*:\s*(\d+)",
    }
    values = {}
    for key, pattern in fields.items():
        match = re.search(pattern, output)
        if not match:
            raise ValueError(f"Could not find '{key}' in output:\n{output}")
        values[key] = match.group(1)
    return ",".join(values[k] for k in ("n_blocks", "preprocessing_time", "kernel_time", "triangles"))


def output_parser_tot(output: str, full_path: str, exe_path: str) -> str:
    if "Empty matrix multiplication result." in output:
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=f"{exe_path} -i {full_path}",
            stderr="Empty matrix multiplication result."
        )
    graph_file = re.search(r'input path:\s*(\S+)', output)
    path = graph_file.group(1) if graph_file else "N/A"
    numbers = re.findall(r'[+-]?\d+\.\d+|[+-]?\d+', output.replace(path, ""))
    return ",".join(numbers)


def output_parser_bbtc(output: str, full_path: str, exe_path: str) -> str:
    graph_file = re.search(r'Graph file:\s*(\S+)', output)
    path = graph_file.group(1) if graph_file else "N/A"
    numbers = re.findall(r'\d+\.\d+|\d+', output.replace(path, ""))
    return ",".join(numbers)


def output_parser_tc(output: str, full_path: str, exe_path: str) -> str:
    return ",".join(list(filter(None, output.split("\n")[1].split(" ")))[1:])


# ---------------------------------------------------------------------------
# Select the profile for the given executable
# ---------------------------------------------------------------------------

exe = args.exe_path.split("/")[-1]

if exe == "tot":
    header = ("exe,graph,nrows,ncols,nnz,extracting_upper_triangle_time,"
              "converting_to_bitmap_time,kernel_time,counting_triangles_time,"
              "triangles,max_memory_consumption")
    make_execution_args = lambda full_path: [args.exe_path, "-i", full_path]
    memory_function = get_max_memory_consumption
    output_parser = output_parser_tot

elif exe == "bbtc":
    header = ("exe,graph,nrows,ncols,nnz,N,algorithm_nnz,n_cuts,n_tasks,n_gpus,"
              "n_workers,triangles,preprocessing_time,malloc_stream_time,"
              "kernel_time,max_memory_consumption")
    make_execution_args = lambda full_path: [args.exe_path, "--graph", full_path, "--repeat", "1"]
    memory_function = get_max_memory_consumption
    output_parser = output_parser_bbtc

elif exe == "tc":
    header = ("exe,graph,nrows,ncols,nnz,n,m,s,a,triangles,prepro_s_time,"
              "gpu_copy_s_time,kernel_time,gpu_total_s_time,cpu_gpu_s_time,"
              "max_memory_consumption")
    make_execution_args = lambda full_path: [args.exe_path, "-m", full_path, "-s", "10"]
    memory_function = get_max_memory_consumption
    output_parser = output_parser_tc

elif "tribit" in exe:
    header = ("exe,graph,nrows,ncols,nnz,n_blocks,preprocessing_time,kernel_time,"
              "triangles,max_memory_consumption")
    make_execution_args = lambda full_path: [args.exe_path, "-i", full_path]
    memory_function = get_max_memory_consumption_pool
    output_parser = output_parser_tribit

else:
    print("Executable is not recognised")
    exit()

if args.denyfile_path:
    with open(args.denyfile_path, 'r') as file:
        denylist = [x.strip() for x in file.readlines()]

# Make sure the folder of the out file exists
out_dir = os.path.dirname(args.out_path)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

iterate = True
counter = 0
with open(args.out_path, 'a+') as file:
    file.seek(0)
    first_line = file.readline().strip()
    if first_line != header:
        file.write(header + "\n")
        file.flush()

    # Walk over all possible files ending with .mtx
    for root, dirs, files in os.walk(args.data_path):
        if not iterate:
            break
        for name in files:
            if not iterate:
                break
            full_path = os.path.join(root, name)
            if full_path.endswith(".mtx"):
                # Optional in-place preprocessing (real/integer/complex -> pattern,
                # keep only first two columns). Off by default: mutates the file.
                if args.preprocess:
                    keep_first_two_columns(full_path)

                # Skip invalid matrices (non-destructive)
                is_valid, nrows, ncols, nnz = mtx_file_is_valid(full_path)
                if not is_valid:
                    continue

                # Skip blacklisted matrices
                if args.denyfile_path:
                    if any(re.search(pattern, name) for pattern in denylist):
                        continue

                # Stop when max_matrices is reached
                if args.max_matrices and counter >= args.max_matrices - 1:
                    iterate = False

                # Stop when dry_run is on
                if args.dry_run:
                    print(counter, name, flush=True)
                    counter += 1
                    continue

                report_id = random_integer()

                try:
                    # Get memory stats
                    if args.get_memory_consumption:
                        max_memory = memory_function(full_path, report_id, make_execution_args)
                    else:
                        max_memory = 0
                    # Get general stats
                    for rep in range(args.n_repetitions):
                        result = subprocess.run(
                            make_execution_args(full_path),
                            capture_output=True,
                            text=True,
                            check=True  # Raise an exception if the program fails
                        )
                        result = output_parser(result.stdout, full_path, args.exe_path)
                        print(counter, name, flush=True)
                        counter += 1
                        # Write results (only when there is no exception)
                        file.write(
                            f"{exe},{full_path.split('/')[-1]},{nrows},{ncols},"
                            f"{nnz},{result},{max_memory}\n"
                        )
                        file.flush()

                except Exception as e:
                    # Try to remove profiler traces, if there were any left
                    for ext in (".nsys-rep", ".sqlite"):
                        try:
                            os.remove(f"report_{report_id}{ext}")
                        except FileNotFoundError:
                            pass

                    print(f"Execution failed for {full_path}", flush=True)
                    print(f"* exception type: {type(e).__name__}", flush=True)
                    print(f"* exception message: {e}", flush=True)
                    if isinstance(e, subprocess.CalledProcessError):
                        print("* stdout:", e.stdout, flush=True)
                        print("* stderr:", e.stderr, flush=True)
                        print("* returncode:", e.returncode, flush=True)

print("Total amount of matrices:", counter)
