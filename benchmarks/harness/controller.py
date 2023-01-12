#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TPP-MLIR Benchmark Controller

    Runs an MLIR kernel multiple times and takes the statistics, comparing to
    a known-good result, checking if any difference is statistically significant.

    Arguments:
     * filename: kernel to run on
     * -n N: Number of times to run (default 1000)
     * -e ENTRY: Entry point name (default "entry")

    Like LLVM's FileCheck, we try to get information from comments on the MLIR
    file (//).

    We can look at RUN lines, for details on how to process the file, like entry
    point, tpp-opt args.

    We also support new ones:
     * BENCH_TOTAL_FLOPS: Number of FP ops in the kernel

    Much of the code was shamelessly stolen from my previous harness:
    https://github.com/Linaro/benchmark_harness/
"""

import os
import sys
import re
import shlex
import argparse

from Logger import Logger
from Execute import Execute
from FileCheckParser import FileCheckParser
from TPPHelper import TPPHelper

class BenchmarkController(object):
    """ Entry point of the benchmark harness"""

    def __init__(self, args, loglevel):
        self.args = args
        self.logger = Logger("controller.bench.controller", loglevel)
        self.helper = TPPHelper(loglevel)
        self.loglevel = loglevel
        # If we're in a git repo, find the base dir, otherwise, this is the base dir
        self.baseDir = self.helper.findGitRoot(os.path.dirname(__file__))
        self.logger.debug(f"Base dir: {self.baseDir}")
        self.programs = self.helper.findTPPProgs(self.baseDir)
        self.output = ''
        self.mean = 0.0
        self.stdev = 0.0
        # Output is always in seconds, we need to convert anyway
        self.unit = "ms" # or 'gflops'

    def verifyArgs(self):
        """ Verify cmd-line and IR file arguments, update defaults, etc """

        # Parse the IR file for user arguments
        self.logger.info("Parsing FileCheck lines, updating arguments")
        fileArgs = FileCheckParser(self.loglevel).parse(args.benchmark_name)

        # Command line arguments have preference
        if (not self.args.n and 'iters' in fileArgs):
            self.args.n = fileArgs['iters']
        if (not self.args.entry and 'entry' in fileArgs):
            self.args.entry = fileArgs['entry']
        if (not self.args.flops and 'flops' in fileArgs):
            self.args.flops = fileArgs['flops']
            self.unit = "gflops"
        if (not self.args.opt_args and 'opt-args' in fileArgs):
            self.args.opt_args = fileArgs['opt-args']

        # Make sure we get all we need
        self.logger.info("Validating arguments")
        if (not self.args.n):
            logger.warning("Number of iterations not found, default to 1000")
            self.args.n = 1000

        if (not self.args.entry):
            logger.error("No entry point defined, bailing")
            return False

        return True

    def run(self):
        """ Run tpp-opt and tpp-run to get the timings """

        irContents = ""
        executor = Execute(self.loglevel)

        # Only run tpp-opt if we have the arguments
        if self.args.opt_args:
            self.logger.info("Running optimiser, to prepare the IR file")
            optCmd = [ self.programs['tpp-opt'], self.args.benchmark_name ]
            optCmd.extend(shlex.split(self.args.opt_args))

            # Run tpp-opt and capture the output IR
            optResult = executor.run(optCmd)
            if optResult.stderr:
                self.logger.error(f"Error executing tpp-opt: {optResult.stderr}")
                return False
            irContents = optResult.stdout
        else:
            # Bypass tpp-opt and just dump the file
            with open(self.args.benchmark_name) as file:
                irContents = file.readall()

        # Actually run the file in benchmark mode, no output
        self.logger.info("Running the kernel with the arguments provided")
        runCmd = [ self.programs['tpp-run'], '-n', str(self.args.n),
                                             '-e', self.args.entry,
                                             '--entry-point-result=void',
                                             '--print=0',
                  ]
        runResult = executor.run(runCmd, irContents)
        if runResult.stderr:
            self.logger.error(f"Error executing tpp-run: {runResult.stderr}")
            return False
        self.output = runResult.stdout

        return True

    def verifyStats(self):
        """ Verify the results, should be in format '( mean, stdev )' """

        if not self.output:
            self.logger.error("Benchmark produced no output, can't verify results")
            return False

        # Parse results (always in seconds, as per timer)
        m = re.search("([\d\.\-e]+), ([\d\.\-e]+)", self.output)
        if m:
            self.mean = float(m.group(1))
            self.stdev = float(m.group(2))
            self.logger.info(f"Mean time: {self.mean*1000} ms +- {self.stdev*1000} ms")
        else:
            self.logger.error("Cannot find mean/stdev in output")
            return False

        # If we asked for flops, we need to convert
        if self.args.flops:
            mean = self.args.flops / self.mean
            stdev = self.args.flops * self.stdev / (self.mean * self.mean)
            self.mean = mean
            self.stdev = stdev
            # We annotate in flops (easier to calculate / compare) but we display in Gflops
            self.mean /= 1e9
            self.stdev /= 1e9
        else:
            # Output is in seconds, but we display in milliseconds
            self.mean *= 1000
            self.stdev *= 1000

        return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TPP-MLIR Benchmark Harness')

    # Required argument: benchmark name (can be a file or a directory)
    parser.add_argument('benchmark_name', type=str,
                        help='MLIR file or directory containing MLIR files')

    # Required, but auto-detected if omitted
    parser.add_argument('-n', type=int,
                        help='Number of times to execute the kernel (checks RUN line)')
    parser.add_argument('-flops', type=float,
                        help='Known number of FP OPs (checks BENCH_EXPECTED_FLOPS line)')
    parser.add_argument('-entry', type=str,
                        help='Name of the entry point (checks RUN line)')
    parser.add_argument('-opt-args', type=str,
                        help='tpp-opt arguments (checks RUN line)')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='The verbosity of logging output')
    parser.add_argument('-q', '--quiet', action='count', default=0,
                        help='Suppress warnings')
    parser.add_argument('-x', '--xsmm', action='count', default=1,
                        help='Turn on TPP optimizations (default)')
    args = parser.parse_args()

    # Creates the logger object
    loglevel = args.verbose - (args.quiet > 0)
    logger = Logger("controller", loglevel)

    # Creates a controller from command line arguments
    controller = BenchmarkController(args, loglevel)

    # Checks all parameters are good
    if (not controller.verifyArgs()):
        logger.error("Argument verification error")
        print('\n\n')
        parser.print_help()
        sys.exit(1)

    # Runs the benchmark
    if (not controller.run()):
        logger.error("Error executing the benchmark")
        sys.exit(1)

    # Checks stats
    if (not controller.verifyStats()):
        logger.error("Error verifying the statistics")
        sys.exit(1)

    # Success prints basic stats
    if args.flops:
        print(f'{(controller.mean):9.3f} +- {(controller.stdev):9.3f} {controller.unit}')
    else:
        print(f'{(controller.mean):3.9f} +- {(controller.stdev):3.9f} {controller.unit}')

