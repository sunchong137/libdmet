import sys
from datetime import datetime

"""
Logger
"""

Level = dict(zip("FATAL ERR RESULT WARNING INFO DEBUG".split(), range(8)))


class Logger:
    """
    - Define Levels of input

    TODO:
    - Implement output functions
    """
    def __init__(self, stdout, verbose, clock = False):
        self.stdout = stdout
        self.verbose = verbose
        self.addclock = clock

    def fatal(self, msg, *args):
        if self.verbose >= Level['FATAL']:
            self.clock()
            flush(self, "FATAL: "+msg, *args)

    def error(self, msg, *args):
        if self.verbose >= Level['ERR']:
            self.clock()
            flush(self, "ERROR: "+msg, *args)
    
    def result(self, msg, *args):
        if self.verbose >= Level['RESULT']:
            self.clock()
            flush(self, "*** " + msg, *args)

    def warning(self, msg, *args):
        if self.verbose >= Level['WARNING']:
            self.clock()
            flush(self, "WARNING: "+msg, *args)
    
    def info(self, msg, *args):
        if self.verbose >= Level['INFO']:
            self.clock()
            flush(self, "****** "+msg, *args)
    
    def debug(self, level, msg, *args):
        if self.verbose >= Level["DEBUG"] + level:
            self.clock()
            flush(self, msg, *args)

    def clock(self):
        if self.addclock:
            self.stdout.write(datetime.now().strftime("%Y %b %d %I:%M:%S") + " ")

def flush(logger, msg, *args):
    logger.stdout.write(msg%args)
    logger.stdout.write('\n')
    logger.stdout.flush()

if __name__ == "__main__":
    log = Logger(sys.stdout, 6, False)
    log.result("Logger Levels: %s", Level)    
    log.warning("Logger Levels: %s", Level)    
    log.info("Logger Levels: %s", Level)
    log.debug(0, "Logger Levels: %s", Level)
    log.debug(1, "Logger Levels: %s", Level)
    log.debug(2, "Logger Levels: %s", Level)
    log.error("Logger Levels: %s", Level)
    log.fatal("Logger Levels: %s", Level)

