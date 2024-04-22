from benchmarks.benchmark_generating_every_sample import benchmark_algorithms
import sys
import argparse


if __name__ == "__main__":
    if len(sys.argv) > 1:
        value = int(sys.argv[1])
    else:
        raise ValueError("Please provide a value as a command line argument.")
    benchmark_algorithms(sample_size=value)