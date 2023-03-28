#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TPP-MLIR Benchmark Driver

    Compares a known good MLIR kernel with a known good reference implementation
    in C++ and XSMM.

    Arguments:
     * directory: must have a JSON file with the configuration of the benchmark

    The directory file must have:
     - Reference C++ file (to be built by CMake and left in the bin directory)
     - At least one MLIR file (multiple if using options)
     - JSON file with the configuration on how to run it

    The JSON format is:
     [
         // This is the run name
         // You can have multiple runs on the same pass
         "64x64x64": {
             "ref": {
                 "type": "C++",
                 "benchmark": "matmul",
                 "flags": [ "--input=64x64x64", "-v" ]
             },
             "xsmm": {
                 "type": "C++",
                 "benchmark": "matmul",
                 "flags": [ "--input=64x64x64", "-v", "-xsmm" ]
             },
             "mlir": {
                 "type": "MLIR",
                 "benchmark": "matmul_64x64x64.mlir",
                 "flags": [ "-v" ]
             },
         },
         "128x256x512":  {
             ...
         },
     ]
"""

import os
import sys
import re
import argparse
import json

sys.path.append('harness')

from Logger import Logger
from Execute import Execute
from TPPHelper import TPPHelper

class Environment(object):
    def __init__(self, args, loglevel):
        self.logger = Logger("driver.env", loglevel)
        helper = TPPHelper(loglevel)
        self.base_dir = os.path.realpath(os.path.dirname(__file__))
        self.root_dir = helper.findGitRoot(self.base_dir)
        self.build_dir = args.build
        if not self.build_dir:
            self.build_dir = self.root_dir
        programs = helper.findTPPProgs(self.build_dir)
        for _, path in programs.items():
            if os.path.exists(path):
                self.bin_dir = os.path.realpath(os.path.dirname(path))
                parent = os.path.join(self.bin_dir, os.path.pardir)
                self.build_dir = os.path.realpath(parent)
                self.lib_dir = os.path.join(self.build_dir, "lib")
                break
        assert(self.build_dir != self.root_dir)
        self.bench_dir = os.path.join(self.root_dir, "benchmarks")
        self.harness = os.path.join(self.bench_dir, "harness", "controller.py")
        self.test_dir = os.path.join(self.bench_dir, "mlir")
        # Pass arguments down to benchmarks, if known
        self.extra_args = list()
        if args.verbose > 0:
            for v in range(args.verbose - args.quiet):
                self.extra_args.append("-v")
        # Set environment variables for dynamic loading (Linux and Mac)
        for path in ["LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"]:
            environ = [os.getenv(path)] if os.getenv(path) else []
            environ.insert(0, self.lib_dir)  # prepend
            os.environ[path] = ":".join(environ)

class BaseRun(object):
    """ Base class for all runs """

    def __init__(self, name, args, env, json, loglevel):
        self.name = name
        self.env = env
        self.args = args
        self.benchmark = json["benchmark"]
        self.flags = json["flags"]
        self.runner = Execute(loglevel)
        self.stdout = ""
        self.stderr = ""

    def run(self):
        # This is mandatory
        return False

class CPPRun(BaseRun):
    """ C++ runs """

    def __init__(self, name, args, env, json, loglevel):
        self.logger = Logger("driver.cpprun", loglevel)
        BaseRun.__init__(self, name, args, env, json, loglevel)
        assert(json["type"] == "C++")
        self.benchmark = os.path.join(env.bin_dir, self.benchmark)

    def run(self):
        command = [self.benchmark]
        if self.flags:
            command.extend(self.flags)
        if self.env.extra_args:
            command.extend(self.env.extra_args)
        res = self.runner.run(command)
        self.stdout = res.stdout
        self.stderr = res.stderr
        return True

class MLIRRun(BaseRun):
    def __init__(self, name, args, env, json, loglevel):
        self.logger = Logger("driver.mlirrun", loglevel)
        BaseRun.__init__(self, name, args, env, json, loglevel)
        assert(json["type"] == "MLIR")
        self.benchmark = os.path.join(env.test_dir, self.benchmark)

    def run(self):
        command = [self.env.harness]
        if self.args.build:
            command.extend(["--build", self.args.build])
        if self.flags:
            command.extend(self.flags)
        if self.env.extra_args:
            command.extend(self.env.extra_args)
        command.append(self.benchmark)
        res = self.runner.run(command)
        self.stdout = res.stdout
        self.stderr = res.stderr
        return True

class Benchmark(object):
    """ A collection of runs """

    def __init__(self, name, args, env, loglevel):
        self.name = name
        self.args = args
        self.env = env
        self.logger = Logger("driver.bench", loglevel)
        self.runs = list()

    def addRun(self, name, json):
        runType = json["type"]
        self.logger.debug(f"Adding {runType} run {name} for {self.name}")
        if runType == "C++":
            self.runs.append(CPPRun(name, self.args, self.env, json, loglevel))
        elif runType == "MLIR":
            self.runs.append(MLIRRun(name, self.args, self.env, json, loglevel))
        else:
            self.logger.error(f"Unknown runner type '{runType}'")
            return False
        return True

    def runAll(self):
        for run in self.runs:
            self.logger.debug(f"Running bench {run.name}")
            if not run.run() and not self.args.ignore_errors:
                return False

        return True

    def getRuns(self):
        return self.runs

class BenchmarkDriver(object):
    """ Detects and runs benchmarks based on JSON configurations """

    def __init__(self, args, loglevel):
        self.logger = Logger("driver.bench.driver", loglevel)
        self.env = Environment(args, loglevel)
        self.loglevel = loglevel
        self.args = args
        if not os.path.exists(self.args.config):
            self.logger.error(f"JSON config '{self.args.config}' does not exist")
            raise SyntaxError("Cannot find JSON config")
        self.benchs = list()

    def scanBenchmarks(self):
        """ Scan directory for JSON file and create a list with all runs """

        # Find and read the JSON file
        self.logger.info(f"Reading up '{self.args.config}'")
        with open(self.args.config) as jsonFile:
            jsonCfg = json.load(jsonFile)

        # Parse and add all runs
        for cfg in jsonCfg:
            if len(cfg.keys()) > 1:
                self.logger.error("List of dict with a single element expected")
                return False

            name = list(cfg.keys())[0]
            runs = cfg[name]
            benchs = Benchmark(name, self.args, self.env, self.loglevel)
            for key, run in runs.items():
                if not benchs.addRun(key, run):
                    return False

            # Append to the benchmarks
            self.benchs.append(benchs)

        return True

    def run(self):
        """ Run tpp-opt and tpp-run to get the timings """

        # Actually run the file in benchmark mode, no output
        self.logger.info("Running the kernels with the arguments provided")

        # Out/Err will be stored in the runs themselves, verify later
        for bench in self.benchs:
            if not bench.runAll():
                if not self.args.ignore_errors:
                    return False

        return True

    def verifyStats(self):
        """ Verify the results, should be in format '( mean, stdev )' """

        for bench in self.benchs:
            print(f"Benchmark: {bench.name}")
            for run in bench.getRuns():
                if not run.stdout:
                    self.logger.error(f"Benchmark {bench.name}, run {run.name} produced no output, can't verify results")
                    self.logger.error(f"Error: {run.stderr}")
                    if not self.args.ignore_errors:
                        return False

                # Clean up output
                stdout = re.sub("\n", "", run.stdout)
                print(f"{run.name:28}: {stdout}")
            print("")

        return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TPP-MLIR Benchmark Harness')

    # Required argument: baseDir name (directory)
    parser.add_argument('-c', '--config', type=str, default="benchmarks.json",
                        help='JSON file containing benchmark configuration')

    # Optional
    parser.add_argument('--build', type=str, default="",
                        help='Path to the build dir')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='The verbosity of logging output')
    parser.add_argument('-q', '--quiet', action='count', default=0,
                        help='Suppress warnings')
    parser.add_argument('--ignore-errors', action='count', default=0,
                        help='Ignore errors and only show the results that work')
    args = parser.parse_args()

    # Creates the logger object
    loglevel = args.verbose - (args.quiet > 0)
    logger = Logger("driver", loglevel)

    # Creates a controller from command line arguments
    driver = BenchmarkDriver(args, loglevel)

    # Detects all benchmarks to run, validates files / args
    if (not driver.scanBenchmarks()):
        logger.error("Error finding benchmarks")
        print('\n\n')
        parser.print_help()
        sys.exit(1)

    # Runs all benchmarks
    if (not driver.run()):
        logger.error("Error executing the benchmarks")
        sys.exit(1)

    # Verify results
    if (not driver.verifyStats()):
        logger.error("Error verifying stats")
        sys.exit(1)
