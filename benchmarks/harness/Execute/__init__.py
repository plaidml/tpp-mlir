# -*- coding: utf-8 -*-
"""
 Execute Commands, return out/err, accepts parser plugins

 Usage:
  out, err = Execute(logger).run(['myapp', '-flag', 'etc'])
"""

import subprocess

class Execute(object):
    """Executes commands, returns out/err"""

    def __init__(self, logger):
        self.logger = logger

    def run(self, program, input=''):
        """Execute Commands, return out/err"""

        if program and not isinstance(program, list):
            raise TypeError("Program needs to be a list of arguments")
        if not program:
            raise ValueError("Need program arguments to execute")

        if self.logger:
            self.logger.debug('Executing: %s' % repr(program))

        # Call the program, capturing stdout/stderr
        result = subprocess.run(program,
                                input=bytes(input, encoding='utf-8'),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)

        # Collect stdout, stderr as UTF-8 strings
        result.stdout = result.stdout.decode('utf-8')
        result.stderr = result.stderr.decode('utf-8')

        # Return
        return result
