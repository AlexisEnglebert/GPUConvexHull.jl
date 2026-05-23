import random
import os

points_size = [10, 100, 1000, 10_000, 100_000, 1_000_000, 10_000_000]

if not os.path.exists("scan_data"):
    os.makedirs("scan_data")

for s in points_size:
    with open(f"scan_data/{s}_input.txt", "w") as file:
        for _ in range(s):
            file.writelines(str(random.randint(0, 1000000000000))+"\n")
        for _ in range(s):
            file.writelines(str(random.randint(0, 2)) + "\n")
