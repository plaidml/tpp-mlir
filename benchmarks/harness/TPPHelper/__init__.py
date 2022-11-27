#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TPP Helper

    Detects paths, libraries, executables, LLVM variables, etc.
"""

import os
import sys
import re
import shlex
import argparse

class TPPHelper(object):
    """ Detects paths, libraries, executables, LLVM variables, etc. """

    def __init__(self, logger):
        self.logger = logger

    def findGitRoot(self, path):
        """ Find the git root directory, if any, or return the input """

        temp = path
        while temp:
            if (os.path.exists(os.path.join(temp, ".git"))):
                return temp
            temp = os.path.abspath(os.path.join(temp, os.pardir))
        return path

    def findTPPProgs(self, baseDir):
        """ Find the necessary TPP programs to run the benchmarks """

        programs = { 'tpp-opt': '', 'tpp-run': '' }
        found = 0
        maxProgs = len(programs.keys())
        for root, dirs, files in os.walk(baseDir):
            for prog in programs.keys():
                if prog in files:
                    programs[prog] = os.path.join(root, prog)
                    self.logger.debug(f"{prog}: {programs[prog]}")
                    found += 1
            if found == maxProgs:
                break

        if found < maxProgs:
            self.logger.error("Cannot find all TPP programs")
            self.logger.error(f"Found: {programs}")
            return {}
        return programs

    def getLLVMVariables(self, programs, baseDir):
        """ Find config values in the LIT config in the build dir """

        # Some variables are not in the LIT config and are known
        nonConfig = {}
        if 'tpp-run' in programs:
            nonConfig['tpplibdir'] = os.path.abspath(
                                        os.path.join(
                                        os.path.dirname(
                                            programs['tpp-run']),
                                            '../lib'))
        # Others we need to find in the config in the build dir
        variables = { 'llvmlibdir': 'config.llvm_lib_dir',
                      'shlibext': 'config.llvm_shlib_ext' }
        # Merge the two and count how many we need to match
        variables.update(nonConfig)
        maxMatches = len(variables.keys()) - len(nonConfig.keys())
        matches = 0
        for root, dirs, files in os.walk(baseDir):
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
                                self.logger.debug(f"Found {key}: {variables[key]}")
                                matches += 1
                        # Leave if found everything
                        if matches == maxMatches:
                            return variables

