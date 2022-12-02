#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TPP-MLIR Benchmark Driver

    Compares a known good MLIR kernel with a known good reference implementation
    in C++ and XSMM.

    Arguments:
     * directory: must have a JSON file with the configuration of the benchmark

    The directory file must have:
     - Reference C++ file (to be linked with the common C++ driver)
     - XSMM C++ file (to be linked with the common C++ driver)
     - At least one MLIR file (multiple if using options)
     - JSON file with the configuration on how to run it

    The JSON format is:
     [
         // This is the run name
         // You can have multiple runs on the same pass
         "64x64x64": {
             "ref": {
                 "type": "C++",
                 "source": "matmul/matmul.cc",
                 "iters" : "1000",
                 "flags": "64x64x64"
             },
             "xsmm": {
                 "type": "C++",
                 "source": "matmul/matmul.cc",
                 "iters" : "1000",
                 "flags": "-x 64x64x64"
             },
             "mlir": {
                 "type": "MLIR",
                 "source": "../../test/Benchmarks/matmul_64x64x64.mlir",
                 "iters" : "1000",
                 "flags": ""
             },
         },
         "128x256x512":  {
             ...
         },
     ]

    The compiler paths/includes/libraries will be detected automatically.
    The source file will be compiled together with the common driver in the
    root of the benchmark directory, so they should only contain the kernel
    logic to be called by the driver.

    The arguments to the MLIR harness will be detected automatically.
"""

import os
import sys
import re
import shlex
import argparse
import json

sys.path.append('harness')

from Logger import Logger
from Execute import Execute
from TPPHelper import TPPHelper

class Environment(object):
    def __init__(self, args, logger):
        helper = TPPHelper(logger)
        self.base_dir = os.path.realpath(os.path.dirname(__file__))
        self.root_dir = helper.findGitRoot(".")
        programs = helper.findTPPProgs(self.root_dir)
        for _, path in programs.items():
            if os.path.exists(path):
                self.bin_dir = os.path.realpath(os.path.dirname(path))
                parent = os.path.join(self.bin_dir, os.path.pardir)
                self.build_dir = os.path.realpath(parent)
                self.lib_dir = os.path.join(self.build_dir, "lib")
                self.inc_path = os.path.join(self.root_dir, "include")
                self.lib_inc_path = os.path.join(self.build_dir, "_deps", "xsmm-src", "include")
                break
        assert(self.build_dir)
        self.bench_dir = os.path.join(self.root_dir, "benchmarks")
        self.harness = os.path.join(self.bench_dir, "harness", "controller.py")
        self.test_dir = os.path.join(self.root_dir, "test", "Benchmarks")
        # Pass arguments down to benchmarks, if known
        self.extra_args = list()
        if args.verbose > 0:
            for v in range(args.verbose - args.quiet):
                self.extra_args.append("-v")

class BaseRun(object):
    """ Base class for all runs """

    def __init__(self, name, env, json, logger):
        self.logger = logger
        self.name = name
        self.env = env
        self.source = json["source"]
        self.iters = json["iters"]
        self.flags = json["flags"]
        self.runner = Execute(logger)
        self.stdout = ""
        self.stderr = ""

    def compile(self):
        # This is optional
        return True

    def run(self):
        # This is mandatory
        return False

    def cleanup(self):
        # This is optional
        return True

class CPPRun(BaseRun):
    """ C++ runs """

    def __init__(self, name, env, json, logger):
        BaseRun.__init__(self, name, env, json, logger)
        assert(json["type"] == "C++")
        source_dir = os.path.dirname(self.source)
        self.binary = os.path.join(source_dir, f"bench_{name}.bin")

    def compile(self):
        command = [
                "clang++",
                "-O3",
                f"-I{self.env.inc_path}",
                f"-I{self.env.lib_inc_path}",
                f"-L{self.env.lib_dir}",
                os.path.join(self.env.build_dir, "libxsmm.a"),
                "-lm",
                self.source,
                "-o",
                self.binary]
        res = self.runner.run(command)
        if res.stderr:
            self.logger.error(f"Error compiling {self.name}")
            self.logger.error(res.stderr)
            return False
        return True

    def run(self):
        command = [self.binary, "-n", self.iters]
        if self.flags:
            command.extend(shlex.split(self.flags))
        if self.env.extra_args:
            command.extend(self.env.extra_args)
        res = self.runner.run(command)
        self.stdout = res.stdout
        self.stderr = res.stderr
        # Extra logs go on stderr
        if self.stderr:
            print(self.stderr, file=sys.stderr)
        return True

    def cleanup(self):
        if os.path.exists(self.binary):
            os.remove(self.binary)
        return True

class MLIRRun(BaseRun):
    def __init__(self, name, env, json, logger):
        BaseRun.__init__(self, name, env, json, logger)
        assert(json["type"] == "MLIR")

    def run(self):
        command = [self.env.harness, "-n", self.iters, self.source]
        if self.flags:
            command.extend(shlex.split(self.flags))
        if self.env.extra_args:
            command.extend(self.env.extra_args)
        res = self.runner.run(command)
        self.stdout = res.stdout
        self.stderr = res.stderr
        # Extra logs go on stderr
        if self.stderr:
            print(self.stderr, file=sys.stderr)
        return True

class Benchmark(object):
    """ A collection of runs """

    def __init__(self, name, env, logger):
        self.name = name
        self.env = env
        self.logger = logger
        self.runs = list()

    def addRun(self, name, json):
        runType = json["type"]
        self.logger.debug(f"Adding {runType} run {name} for {self.name}")
        if runType == "C++":
            self.runs.append(CPPRun(name, self.env, json, logger))
        elif runType == "MLIR":
            self.runs.append(MLIRRun(name, self.env, json, logger))
        else:
            self.logger.error(f"Unknown runner type '{runType}'")
            return False
        return True

    def runAll(self):
        for run in self.runs:
            self.logger.debug(f"Running bench {run.name}")
            if not run.compile():
                return False
            if not run.run():
                return False

        return True

    def getRuns(self):
        return self.runs

class BenchmarkDriver(object):
    """ Detects and runs benchmarks based on JSON configurations """

    def __init__(self, args, logger):
        self.logger = logger
        self.env = Environment(args, logger)
        self.config = args.config
        if not os.path.exists(self.config):
            self.logger.error(f"JSON config '{self.config}' does not exist")
            raise SyntaxError("Cannot find JSON config")
        self.benchs = list()

    def scanBenchmarks(self):
        """ Scan directory for JSON file and create a list with all runs """

        # Find and read the JSON file
        self.logger.info(f"Reading up '{self.config}'")
        with open(self.config) as jsonFile:
            jsonCfg = json.load(jsonFile)

        # Parse and add all runs
        for cfg in jsonCfg:
            if len(cfg.keys()) > 1:
                self.logger.error("List of dict with a single element expected")
                return False

            name = list(cfg.keys())[0]
            runs = cfg[name]
            benchs = Benchmark(name, self.env, self.logger)
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
                    return False

                # Clean up output
                stdout = re.sub("\n", "", run.stdout)
                print(f"{run.name:20}: {stdout}")
            print("")

        return True

    def cleanup(self):
        for bench in self.benchs:
            for run in bench.getRuns():
                run.cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TPP-MLIR Benchmark Harness')

    # Required argument: baseDir name (directory)
    parser.add_argument('-c', '--config', type=str, default="benchmarks.json",
                        help='JSON file containing benchmark configuration')

    # Optional
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='The verbosity of logging output')
    parser.add_argument('-q', '--quiet', action='count', default=0,
                        help='Suppress warnings')
    parser.add_argument('-k', '--keep', action='count', default=0,
                        help='Keep binaries after execution')
    args = parser.parse_args()

    # Creates the logger object
    logger = Logger(__name__, parser, args.verbose - (args.quiet > 0))

    # Creates a controller from command line arguments
    driver = BenchmarkDriver(args, logger)

    # Detects all benchmarks to run, validates files / args
    if (not driver.scanBenchmarks()):
        logger.error("Error finding benchmarks", print_help=True)
        sys.exit(1)

    # Runs all benchmarks
    if (not driver.run()):
        logger.error("Error executing the benchmarks")
        sys.exit(1)

    # Verify results
    if (not driver.verifyStats()):
        logger.error("Error verifying stats")
        sys.exit(1)

    # Cleanup
    if not args.keep:
        driver.cleanup()
