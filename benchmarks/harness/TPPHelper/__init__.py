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

from Logger import Logger


class TPPHelper(object):
    """Detects paths, libraries, executables, LLVM variables, etc."""

    def __init__(self, loglevel):
        self.logger = Logger("tpp.helper", loglevel)

    def findGitRoot(self, path):
        """Find the git root directory, if any, or return the input"""

        temp = path
        while temp:
            if os.path.exists(os.path.join(temp, ".git")):
                return temp
            temp = os.path.abspath(os.path.join(temp, os.pardir))
        return path

    def findTPPProgs(self, baseDir):
        """Find the necessary TPP programs to run the benchmarks"""

        programs = {"tpp-opt": "", "tpp-run": ""}
        found = 0
        maxProgs = len(programs.keys())
        for root, dirs, files in os.walk(baseDir, followlinks=True):
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
