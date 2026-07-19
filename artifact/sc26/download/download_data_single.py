from datetime import datetime
import subprocess
import argparse
import ssgetpy
import os

start_time = datetime.now()
print(f"Begin: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required = True) 
parser.add_argument('--output', type=str, required = True) 
args = parser.parse_args()

with open(args.input, 'r') as file:
    names = [line.strip().replace('.mtx', '') for line in file.readlines() if line.strip()]

print(f'{len(names)} graphs detected')

os.makedirs(args.output, exist_ok=True)

for name in names:
    # Get search results
    if '/' in name:
        group, name = name.split('/')[-2:]
        results = ssgetpy.search(name=name, group=group)
    else:
        results = ssgetpy.search(name=name)

    # Nothing was found
    if not results:
        print(f"NOT FOUND: {name}")
        continue

    exact = [r for r in results if r.name == name]
    # Results were found, but none is an exact match
    if not exact:
        print(f"NOT FOUND (no exact match): {name}")
        continue

    match = exact[0]

    # Skip if already existing
    extracted_path = f"{args.output}/{match.name}"
    if os.path.exists(extracted_path):
        print(f"SKIPPING (already exists): {match.name}")
        continue

    # Build URL and download
    url = f"https://sparse.tamu.edu/MM/{match.group}/{match.name}.tar.gz"
    out_path = f"{args.output}/{match.name}.tar.gz"
    print(f"Downloading {match.name} from {url}")
    result = subprocess.run(
        ["curl", "-fL", "-o", out_path, url],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"FAILED: {name} — {result.stderr.strip()}")
    else:
        # Extract and remove tar file 
        subprocess.run(["tar", "-xzf", out_path, "-C", f"{args.output}"])
        os.remove(out_path)

end_time = datetime.now()
print(f"End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Elapsed: {end_time - start_time}")


