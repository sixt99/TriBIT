from tabulate import tabulate
import argparse 
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required = True)
parser.add_argument('--output', type=str)
args = parser.parse_args()

with open(args.file) as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

table = tabulate(rows, headers=header, tablefmt="simple")

if args.output:
    with open(args.output, "w") as f:
        f.write(table)

print(table)
