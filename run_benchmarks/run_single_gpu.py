import pandas as pd
import subprocess
import argparse
import tempfile
import sqlite3
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('--exe_path', type=str, required = True)
parser.add_argument('--data_path', type=str, required = True)
parser.add_argument('--denyfile_path', type=str)
parser.add_argument('--out_path', type=str, default = 'results.csv')
parser.add_argument('--n_repetitions', type=int, default = 1)
parser.add_argument('--dry_run', action='store_true')
parser.add_argument('--get_memory_consumption', action='store_true')
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

def get_sqlite3_connection(full_path, report_name):
    nsys_launch_args = [
        "nsys", "profile",
        "--trace", "cuda",
        "--cuda-memory-usage", "true",
        "--export", "sqlite",
        "-f", "true",
        "-o", report_name 
    ] + make_execution_args(full_path)
    result = subprocess.run(
        nsys_launch_args,
        capture_output=False,
        text=False,
        check=True # raises an exception if the program fails
    )
    os.remove(report_name + ".nsys-rep")
    conn = sqlite3.connect(report_name + ".sqlite")
    return conn

def get_max_memory_consumption_pool(full_path, report_id):
    report_name = f"report_{report_id}"
    conn = get_sqlite3_connection(full_path, report_name)
    df = pd.read_sql(
        '''
            SELECT localMemoryPoolUtilizedSize
            FROM CUDA_GPU_MEMORY_USAGE_EVENTS;
        ''', conn)
    static_memory = pd.read_sql(
        '''
            SELECT SUM(bytes)
            FROM CUDA_GPU_MEMORY_USAGE_EVENTS
            WHERE memKind = 5;
        ''', conn).iloc[0, 0] or 0 # if this is none, put a 0 instead
    conn.close()
    os.remove(report_name + ".sqlite")
    #peak = int(df.dropna().to_numpy().flatten()[2:].max())  # skip first two dummy allocations
    peak = int(df.dropna().to_numpy().flatten().max())
    return peak, static_memory

def output_parser_tribit(output: str) -> str:
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

    return ",".join(values[k] for k in ("graph", "n_blocks", "preprocessing_time", "kernel_time", "triangles"))

exe = args.exe_path.split("/")[-1]
if "tribit" in exe:
    header = "exe,graph,nrows,ncols,nnz,n_blocks,preprocessing_time,kernel_time,triangles,max_memory_consumption,static_memory"
    make_execution_args = lambda full_path: [args.exe_path, "-i", full_path]
    memory_function = get_max_memory_consumption_pool
    output_parser = output_parser_tribit

else:
    print("Executable is not recognised")
    exit()

if args.denyfile_path:
    with open(args.denyfile_path, 'r') as file:
        denylist = [x.strip() for x in file.readlines()]

counter = 0
with open(args.out_path, 'a+') as file:
    file.seek(0)
    first_line = file.readline().replace("\n", "")
    if first_line != header:
        file.write(header + "\n")
        file.flush()

    # Walk over all possible files ending with .mtx
    for root, dirs, files in os.walk(args.data_path):
        for name in files:
            full_path = os.path.join(root, name)
            if full_path.endswith(".mtx"):
                # Skip invalid matrices
                is_valid, nrows, ncols, nnz = mtx_file_is_valid(full_path)
                if not is_valid:
                    continue
                # Skip blacklisted matrices
                if args.denyfile_path:
                    if any(re.search(pattern, name) for pattern in denylist):
                        continue
                # Print name of valid matrix
                if (args.dry_run):
                    print(name, flush = True)
                    counter += 1
                    continue
                
                report_id = random_integer()
                try:
                    # Get memory stats
                    if args.get_memory_consumption:
                        max_memory, static = memory_function(full_path, report_id)
                    else:
                        max_memory = 0
                        static = 0
                    # Get general stats
                    for rep in range(args.n_repetitions):
                        result = subprocess.run(
                            make_execution_args(full_path),
                            capture_output=True,
                            text=True,
                            check=True # Raise an exception if the program fails
                        )
                        result = output_parser(result.stdout)
                        print(name, flush = True)
                        counter += 1
                        # Write results (only when there is no exception)
                        file.write(f"{args.exe_path},{full_path.split('/')[-1]},{nrows},{ncols},{nnz},{result},{max_memory},{static}\n")
                        file.flush()

                except Exception as e:
                    # Try to remove profiler traces, if there were any left
                    try:
                        os.remove(f"report_{report_id}.nsys-rep")
                        os.remove(f"report_{report_id}.sqlite")
                    except FileNotFoundError:
                        pass

                    print(f"Execution failed for {full_path}")
                    print(f"* exception type: {type(e).__name__}")
                    print(f"* exception message: {e}")
                    if isinstance(e, subprocess.CalledProcessError):
                        print("* stdout:", e.stdout)
                        print("* stderr:", e.stderr)
                        print("* returncode:", e.returncode)

print("Total amount of matrices:", counter)
