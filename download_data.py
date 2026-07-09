import ssgetpy
import subprocess
import os

with open('graphs.txt', 'r') as file:
    names = [line.strip().replace('.mtx', '') for line in file.readlines() if line.strip()]

print(f'{len(names)} graphs detected')

os.makedirs('data', exist_ok=True)

for name in names:
    results = ssgetpy.search(name=name)
    if not results:
        print(f"NOT FOUND: {name}")
        continue
    match = results[0]

    # Skip if already existing
    extracted_path = f"data/{match.name}"
    if os.path.exists(extracted_path):
        print(f"SKIPPING (already exists): {match.name}")
        continue

    url = f"https://sparse.tamu.edu/MM/{match.group}/{match.name}.tar.gz"
    out_path = f"data/{match.name}.tar.gz"

    print(f"Downloading {match.name} from {url}")
    result = subprocess.run(
        ["curl", "-fL", "-o", out_path, url],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"FAILED: {name} — {result.stderr.strip()}")
    else:
        # extract right away
        subprocess.run(["tar", "-xzf", out_path, "-C", "data"])
        os.remove(out_path)
