# -*- coding: utf-8 -*-
"""
 Execute Commands, return out/err, accepts parser plugins

 Usage:
  out, err = Execute(logger).run(['myapp', '-flag', 'etc'])
"""

import subprocess

from Logger import Logger

class Execute(object):
    """Executes commands, returns out/err"""

    def __init__(self, loglevel):
        self.logger = Logger("execute", loglevel)

    def run(self, program, input=''):
        """Execute Commands, return out/err"""

        if program and not isinstance(program, list):
            raise TypeError("Program needs to be a list of arguments")
        if not program:
            raise ValueError("Need program arguments to execute")

        if self.logger:
            self.logger.debug(f"Executing: {' '.join(program)}")

        # Call the program, capturing stdout/stderr
        result = subprocess.run(program,
                                input=input if input else None,
                                capture_output=True,
                                encoding="utf-8")

        # Collect stdout, stderr as UTF-8 strings
        result.stdout = str(result.stdout)
        result.stderr = str(result.stderr)

        # Return
        return result
