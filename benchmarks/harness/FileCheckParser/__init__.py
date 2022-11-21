# -*- coding: utf-8 -*-
"""
 Parse IR files for comments indicating metadata that we can actually use.

 Usage:
  params = FileCheckParser(self.logger).parse('kernel.mlir')

 Params: A dictionary containing the following fields, if found
    - 'iters': Iterations, from a RUN line
    - 'entry': Kernel function name, from a RUN line
    - 'libs': Shared library argument, from a RUN line
    - 'expected_mean': From a BENCH_EXPECTED_MEAN line
    - 'expected_stdev': From a BENCH_EXPECTED_STDEV line
"""

import re

class FileCheckParser(object):
    """Parsers IR files for FileCheck lines to extract informatio
       about the execution of the kernel"""

    def __init__(self, logger=None):
        self.logger = logger
        # FileCheck line style
        self.runRE = re.compile("^\/\/\s*RUN: (.*)$");
        self.expectMeanRE = re.compile("^\/\/\s*BENCH_EXPECT_MEAN: ([\d\.\-e]+)")
        self.expectStdevRE = re.compile("^\/\/\s*BENCH_EXPECT_STDEV: ([\d\.\-e]+)")
        # Arguments in the RUN lines for tpp-opt
        self.optArgs = re.compile("tpp-opt (%\w+)\s+(.*?)\s*\|")
        # Arguments in the RUN lines for tpp-run
        self.entryRE = re.compile("-e\s+(\w+)\s")
        self.itersRE = re.compile("-n\s+(\d+)\s")
        self.libsRE = re.compile("-shared-libs=([^\s]+)\s")
        # All matches
        self.result = {}

    def _parseLines(self, file):
        """ Scan the file for FileCheck lines and update the results cache """

        runLine = ""
        for line in file.readlines():
            # First the easy ones: mean/stdev
            m = self.expectMeanRE.match(line)
            if m:
                self.logger.debug("MEAN line detected: " + m.group(1))
                if 'mean' in self.result:
                    self.logger.warning("Multiple mean lines detected, using last one")
                self.result['mean'] = float(m.group(1))

            m = self.expectStdevRE.match(line)
            if m:
                self.logger.debug("STDEV line detected: " + m.group(1))
                if 'stdev' in self.result:
                    self.logger.warning("Multiple stdev lines detected, using last one")
                self.result['stdev'] = float(m.group(1))

            # Now, concatenate all RUN lines, to make sure we can match
            # arguments through line breaks
            m = self.runRE.match(line)
            if m:
                runLine += m.group(1)

        # If we found any RUN line, clean it up
        if runLine:
            runLine = re.sub('\\\\', '', runLine)
        self.logger.debug("RUN line detected: " + runLine)

        # Now we match the remaining args in the RUN lines
        m = self.optArgs.search(runLine)
        if m:
            self.result['opt-args'] = m.group(2)
            self.logger.debug("Opt args detected: " + m.group(2))

        m = self.entryRE.search(runLine)
        if m:
            self.result['entry'] = m.group(1)
            self.logger.debug("Entry point detected: " + m.group(1))
        else:
            self.logger.info("Did not find the entry point argument in RUN lines")

        m = self.itersRE.search(runLine)
        if m:
            self.result['iters'] = int(m.group(1))
            self.logger.debug("Number of iterations detected: " + m.group(1))
        else:
            self.logger.info("Did not find the iterations argument in RUN lines")

        m = self.libsRE.search(runLine)
        if m:
            self.result['shared-libs'] = m.group(1)
            self.logger.debug("Shared libraries detected: " + m.group(1))
        else:
            self.logger.info("Did not find the shared libs argument in RUN lines")

    def parse(self, filename):
        """Parses an IR file, returns a dictsionary with the data found"""

        try:
            with open(filename) as file:
                self._parseLines(file)
        except IOError as err:
            self.logger.error("Cannot open file '" + filename + "': " + err.strerror)
            return {}
        except ValueError as err:
            self.logger.error("Cannot convert string into int/float: " + err.strerror)
            return {}
        except NameError as err:
            self.logger.error("Name error while parsing IR file: " + err.args)
            return {}
        except Exception as err:
            self.logger.error("Uncaught error while parsing IR file: " + err.strerror)
            return {}

        # Return
        return self.result
