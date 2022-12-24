import sys

file_name = sys.argv[1]
pids = []

with open(file_name, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        split = list(filter(lambda x: len(x) > 0, line.split(" ")))
        pid = split[13]
        pids.append(pid)

print("\n".join(pids))