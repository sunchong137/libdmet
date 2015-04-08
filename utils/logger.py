import sys
from datetime import datetime

"""
Logger
"""

Level = dict(zip("FATAL ERR RESULT WARNING INFO DEBUG0 DEBUG1 DEBUG2".split(), range(8)))

stdout = sys.stdout
verbose = "INFO"
clock = False

def fatal(msg, *args):
    if verbose >= Level['FATAL']:
        __clock()
        flush("  FATAL ", msg, *args, indent = clock * 19)

def fassert(cond, msg, *args):
    if not cond:
        fatal(msg, *args)
        raise Exception

def error(msg, *args):
    if verbose >= Level['ERR']:
        __clock()
        flush("  ERROR ", msg, *args, indent = clock * 19)

def eassert(cond, msg, *args):
    if not cond:
        error(msg, *args)
        raise Exception

def result(msg, *args):
    if verbose >= Level['RESULT']:
        __clock()
        flush("******* ", msg, *args, indent = clock * 19)

def warning(msg, *args):
    if verbose >= Level['WARNING']:
        __clock()
        flush("WARNING ", msg, *args, indent = clock * 19)

def check(cond, msg, *args):
    if not cond:
        warning(msg, *args)

def info(msg, *args):
    if verbose >= Level['INFO']:
        __clock()
        flush("   INFO ", msg, *args, indent = clock * 19)

def debug(level, msg, *args):
    if verbose >= Level["DEBUG0"] + level:
        __clock()
        flush("  DEBUG " + "  " * level, msg, *args, indent = clock * 19)

def flush(msgtype, msg, *args, **kwargs):
    indent = 0
    if len(msg) > 0:
      if "indent" in kwargs:
          indent = kwargs["indent"]

      __msg = (msg % args).split('\n')
      __msg = map(lambda line: msgtype + line, __msg)
      __msg = ("\n" + " " * indent) .join(__msg)
      stdout.write(__msg)
    
    stdout.write('\n')
    stdout.flush()

def __clock():
    if clock:
        stdout.write(datetime.now().strftime("%y %b %d %I:%M:%S") + " ")

if __name__ == "__main__":
    result("Logger Levels: %s", Level)    
    warning("Logger Levels: %s", Level)    
    info("Logger Levels: %s", Level)
    debug(0, "Logger Levels: %s", Level)
    debug(1, "Logger Levels: %s", Level)
    debug(2, "Logger Levels: %s", Level)
    error("Logger Levels: %s", Level)
    fatal("Logger Levels: %s", Level)

