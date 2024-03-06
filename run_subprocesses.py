import subprocess
import re
import os
import pandas as pd
import time

if __name__ == "__main__":
    results = []
    number_of_executions = 3
    for i in range(number_of_executions):
        samples = [5, 10]
        for sample in samples:
            p = subprocess.Popen(["python", "run_benchmark.py", str(sample)], stdout=subprocess.PIPE)
            out = p.stdout.read()
            res = re.findall(r'\(.*?\)', str(out))
            for r in res:
                r = r.replace("(", "").replace(")", "").replace("'", "")
                r = r.split(", ")
                r[1] = int(r[1])
                r[2] = float(r[2])
                r[3] = float(r[3])
                results.append((r[0], r[1], i, r[2], r[3]))
            print(res)


    dfresults = pd.DataFrame(results, columns=["Algorithm", "Samples", "Execution", "Generation Time", "Sample Time"])
    try:
         os.mkdir("./benchmarks/results")
    except OSError as error:
         print(error)
    dfresults.to_csv(f"./benchmarks/results/benchmark_1_{time.time()}.csv", index=False)