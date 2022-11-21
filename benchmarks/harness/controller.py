#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TPP-MLIR Benchmark Driver

    Runs an MLIR kernel multiple times and takes the statistics, comparing to
    a known-good result, checking if any difference is statistically significant.

    Arguments:
     * filename: kernel to run on
     * -n N: Number of times to run (default 1000)
     * -e ENTRY: Entry point name (default "entry")
     * -mean MEAN: Expected mean
     * -stdev STDEV: Expected standard deviation
     * -shared-libs=PATH: MLIR/TPP runtime paths

    Like LLVM's FileCheck, we try to get information from comments on the MLIR
    file (//).

    We can look at RUN lines, for details on how to process the file, like entry
    point, shared-libs, tpp-opt args.

    We also support new ones:
     * BENCH_EXPECTED_MEAN: FP -> Expected value for mean
     * BENCH_EXPECTED_STDIV: FP -> Expected value for standard deviation

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

class BenchmarkController(object):
    """ Entry point of the benchmark harness"""

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        # If we're in a git repo, find the base dir, otherwise, this is the base dir
        self.baseDir = self._findGitRoot(os.path.dirname(__file__))
        self.logger.debug("Base dir: " + self.baseDir)
        self.programs = self._findTPPProgs()
        self.variables = self._getLLVMVariables(self.baseDir)
        self.output = ''
        self.mean = 0.0
        self.stdev = 0.0

    def _findGitRoot(self, path):
        """ Find the git root directory, if any, or return the input """

        temp = path
        while temp:
            if (os.path.exists(os.path.join(temp, ".git"))):
                return temp
            temp = os.path.abspath(os.path.join(temp, os.pardir))
        return path

    def _findTPPProgs(self):
        """ Find the necessary TPP programs to run the benchmarks """

        programs = { 'tpp-opt': '', 'tpp-run': '' }
        found = 0
        maxProgs = len(programs.keys())
        for root, dirs, files in os.walk(self.baseDir):
            for prog in programs.keys():
                if prog in files:
                    programs[prog] = os.path.join(root, prog)
                    self.logger.debug(prog + ": " + programs[prog])
                    found += 1
            if found == maxProgs:
                break

        if found < maxProgs:
            self.logger.error("Cannot find all TPP programs")
            self.logger.error("Found: " + programs)
            return {}
        return programs

    def _getLLVMVariables(self, path):
        """ Find config values in the LIT config in the build dir """

        # Some variables are not in the LIT config and are known
        nonConfig = {}
        if 'tpp-run' in self.programs:
            nonConfig['tpplibdir'] = os.path.abspath(
                                        os.path.join(
                                        os.path.dirname(
                                            self.programs['tpp-run']),
                                            '../lib'))
        # Others we need to find in the config in the build dir
        variables = { 'llvmlibdir': 'config.llvm_lib_dir',
                      'shlibext': 'config.llvm_shlib_ext' }
        # Merge the two and count how many we need to match
        variables.update(nonConfig)
        maxMatches = len(variables.keys()) - len(nonConfig.keys())
        matches = 0
        for root, dirs, files in os.walk(self.baseDir):
            if "lit.site.cfg.py" in files:
                filename = os.path.join(root, "lit.site.cfg.py")
                with open(filename) as file:
                    for line in file.readlines():
                        for key, val in variables.items():
                            # Skip the ones found already
                            if not val.startswith('config.'):
                                continue
                            # Find the config and replace with the value
                            m = re.match(val + " = \"([^\\\"]+)\"", line)
                            if m:
                                variables[key] = m.group(1)
                                self.logger.debug("Found " + key + ": " + variables[key])
                                matches += 1
                        # Leave if found everything
                        if matches == maxMatches:
                            return variables

    def verifyArgs(self):
        """ Verify cmd-line and IR file arguments, update defaults, etc """

        # Parse the IR file for user arguments
        self.logger.info("Parsing FileCheck lines, updating arguments")
        fileArgs = FileCheckParser(self.logger).parse(args.benchmark_name)

        # Command line arguments have preference
        if (not self.args.n and 'iters' in fileArgs):
            self.args.n = fileArgs['iters']
        if (not self.args.entry and 'entry' in fileArgs):
            self.args.entry = fileArgs['entry']
        if (not self.args.shared_libs and 'shared-libs' in fileArgs):
            self.args.shared_libs = fileArgs['shared-libs']
        if (not self.args.mean and 'mean' in fileArgs):
            self.args.mean = fileArgs['mean']
        if (not self.args.stdev and 'stdev' in fileArgs):
            self.args.stdev = fileArgs['stdev']
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

        if (not self.args.shared_libs):
            logger.warning("No shared library path, benchmark may not run correctly")
        else:
            # Make sure the line doesn't have %vars% which can't be resolved
            m = re.search("%", self.args.shared_libs)
            if m:
                for key, val in self.variables.items():
                    self.args.shared_libs = re.sub("%" + key, val, self.args.shared_libs)
                self.logger.debug("Shared libraries updated: " + self.args.shared_libs)

            # Check again, to make sure we replaced everything
            m = re.search("%", self.args.shared_libs)
            if m:
                self.logger.error("Shared libs argument contain %variables, won't be able to resolve")
                return False

        # We don't need mean/stdev for running, but we also can't judge results without them
        if (not self.args.mean or not self.args.stdev):
            self.logger.warning("Need both mean/stdev to compare against a known good value")

        return True

    def run(self):
        """ Run tpp-opt and tpp-run to get the timings """

        irContents = ""
        executor = Execute(self.logger)

        # Only run tpp-opt if we have the arguments
        if self.args.opt_args:
            self.logger.info("Running optimiser, to prepare the IR file")
            optCmd = [ self.programs['tpp-opt'], self.args.benchmark_name ]
            optCmd.extend(shlex.split(self.args.opt_args))

            # Run tpp-opt and capture the output IR
            optResult = executor.run(optCmd)
            if optResult.stderr:
                self.logger.error("Error executing tpp-opt: " + optResult.stderr)
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
                                             '--shared-libs=' + self.args.shared_libs,
                                             '--print=0',
                  ]
        runResult = executor.run(runCmd, irContents)
        if runResult.stderr:
            self.logger.error("Error executing tpp-run: " + runResult.stderr)
            return False
        self.output = runResult.stdout

        return True

    def verifyStats(self):
        """ Verify the results, should be in format '( mean, stdev )' """

        if not self.output:
            self.logger.error("Benchmark produced no output, can't verify results")
            return False

        # Parse results
        m = re.search("([\d\.\-e]+), ([\d\.\-e]+)", self.output)
        if m:
            self.mean = float(m.group(1))
            self.stdev = float(m.group(2))
            self.logger.info("Mean time: " + str(self.mean) + "s (" + str(self.stdev) + "s)")
        else:
            self.logger.error("Cannot find mean/stdev in output")
            return False

        # Check against expected output, if any
        self.logger.info("Validate statistics against expected values")
        if self.args.mean and self.args.stdev:
            em = float(self.args.mean)
            es = float(self.args.stdev)
            if self.mean > (em - es) and self.mean < (em + es):
                self.logger.info("Result mean compatible with expected")
            else:
                self.logger.error("Result mean not compatible with expected")
                return False

            if self.stdev > es:
                self.logger.error("Result deviation too large: " + str(self.stdev) + " > " + str(es))
                return False

        return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TPP-MLIR Benchmark Harness')

    # Required argument: benchmark name (can be a file or a directory)
    parser.add_argument('benchmark_name', type=str,
                        help='MLIR file or directory containing MLIR files')

    # Required, but auto-detected if omitted
    parser.add_argument('-n', type=int,
                        help='Number of times to execute the kernel (checks BENCH_N line)')
    parser.add_argument('-mean', type=float,
                        help='Expected mean to compare to (checks BENCH_EXPECTED_MEAN line)')
    parser.add_argument('-stdev', type=float,
                        help='Expected stdev to compare to (checks BENCH_EXPECTED_STDEV line)')
    parser.add_argument('-entry', type=str,
                        help='Name of the entry point (checks RUN line)')
    parser.add_argument('-shared-libs', type=str,
                        help='Path of the runtime libraries (checks RUN line)')
    parser.add_argument('-opt-args', type=str,
                        help='tpp-opt arguments (checks RUN line)')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='The verbosity of logging output')
    parser.add_argument('-q', '--quiet', action='count', default=0,
                        help='Suppress warnings')
    args = parser.parse_args()

    # Creates the logger object
    logger = Logger(__name__, parser, args.verbose - (args.quiet > 0))

    # Creates a controller from command line arguments
    controller = BenchmarkController(args, logger)

    # Checks all parameters are good
    if (not controller.verifyArgs()):
        logger.error("Argument verification error", print_help=True)
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
    print(f'{args.benchmark_name}: {(controller.mean*1000):3.6f} ms ({(controller.stdev*1000):3.6f} ms)')

