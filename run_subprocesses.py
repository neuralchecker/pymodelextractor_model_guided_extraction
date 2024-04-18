import subprocess
import re
import os
import pandas as pd
import datetime

if __name__ == "__main__":
    results = []
    number_of_executions = 6
    for i in range(number_of_executions):
        samples = [20, 40, 160]
        for sample in samples:
            #this is to run outlines in another process so the cache doesnt affect the results
            p = subprocess.Popen(["python", "run_benchmark.py", str(sample)], stdout=subprocess.PIPE)
            out = p.stdout.read()
            res = re.findall(r'\(.*?\)', str(out))
            for r in res:
                r = r.replace("(", "").replace(")", "").replace("'", "")
                r = r.split(", ")
                r[1] = int(r[1])
                r[2] = float(r[2])
                r[3] = float(r[3])
                if i != 0:
                    results.append((r[0], r[1], i, r[2], r[3]))
            print(res)


    dfresults = pd.DataFrame(results, columns=["Algorithm", "Samples", "Execution", "Generation Time", "Sample Time"])
    try:
         os.mkdir("./benchmarks/results")
    except OSError as error:
         print(error)
    dfresults.to_csv('./benchmarks/results/' +
                     datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")+'.csv', index=False)