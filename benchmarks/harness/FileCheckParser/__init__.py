# -*- coding: utf-8 -*-
"""
 Parse IR files for comments indicating metadata that we can actually use.

 Usage:
  params = FileCheckParser(self.logger).parse('kernel.mlir')

 Params: A dictionary containing the following fields, if found
    - 'iters': Iterations, from a RUN line
    - 'entry': Kernel function name, from a RUN line
    - 'libs': Shared library argument, from a RUN line
    - 'flops': From a BENCH_TOTAL_FLOPS line
"""

import re

from Logger import Logger


class FileCheckParser(object):
    """Parsers IR files for FileCheck lines to extract informatio
    about the execution of the kernel"""

    def __init__(self, loglevel):
        self.logger = Logger("filecheck.parser", loglevel)
        # FileCheck line style
        self.runRE = re.compile(r"^\/\/\s*RUN: (.*)$")
        self.flopsRE = re.compile(r"^\/\/\s*BENCH_TOTAL_FLOPS: ([\d\.\-e]+)")
        # Arguments in the RUN lines for tpp-opt
        self.optArgs = re.compile(r"tpp-opt (%\w+)\s+(.*?)\s*\|")
        # Arguments in the RUN lines for tpp-run
        self.entryRE = re.compile(r"-e\s+(\w+)\s")
        self.itersRE = re.compile(r"-n\s+(\d+)\s")
        self.libsRE = re.compile(r"-shared-libs=([^\s]+)")
        # All matches
        self.result = {}

    def _parseLines(self, file):
        """Scan the file for FileCheck lines and update the results cache"""

        runLine = ""
        for line in file:
            # First the easy one: flops
            m = self.flopsRE.match(line)
            if m:
                self.logger.debug(f"FLOPS line detected: {m.group(1)}")
                if "flops" in self.result:
                    self.logger.warning(
                        "Multiple flops lines detected, using last one"
                    )
                self.result["flops"] = float(m.group(1))
                continue

            # Now, concatenate all RUN lines, to make sure we can match
            # arguments through line breaks
            m = self.runRE.match(line)
            if m:
                runLine += m.group(1)

        # If we found any RUN line, clean it up
        if runLine:
            runLine = re.sub("\\\\", "", runLine)
            self.logger.debug(f"RUN line detected: {runLine}")
        else:
            self.logger.warning("No RUN line detected")

        # Now we match the remaining args in the RUN lines
        m = self.optArgs.search(runLine)
        if m:
            self.result["opt-args"] = m.group(2)
            self.logger.debug(f"Opt args detected: {m.group(2)}")

        m = self.entryRE.search(runLine)
        if m:
            self.result["entry"] = m.group(1)
            self.logger.debug(f"Entry point detected: {m.group(1)}")
        else:
            self.logger.info(
                "Did not find the entry point argument in RUN lines"
            )

        m = self.itersRE.search(runLine)
        if m:
            self.result["iters"] = int(m.group(1))
            self.logger.debug(f"Number of iterations detected: {m.group(1)}")
        else:
            self.logger.info(
                "Did not find the iterations argument in RUN lines"
            )

        m = self.libsRE.search(runLine)
        if m:
            self.result["shared-libs"] = m.group(1)
            self.logger.debug(f"Shared libraries detected: {m.group(1)}")

    def parse(self, input):
        """Parses an IR file, returns a dictsionary with the data found"""

        try:
            if len(input) > 1:
                self._parseLines(input.split("\n"))
            else:
                with open(input) as file:
                    self._parseLines(file)
        except IOError as err:
            self.logger.error(f"Cannot open file '{input}': {err.strerror}")
            return {}
        except ValueError as err:
            self.logger.error(
                f"Cannot convert string into int/float: {err.strerror}"
            )
            return {}
        except NameError as err:
            self.logger.error(f"Name error while parsing IR file: {err.args}")
            return {}
        except Exception as err:
            self.logger.error(
                f"Uncaught error while parsing IR file: {err.strerror}"
            )
            return {}

        # Return
        return self.result
