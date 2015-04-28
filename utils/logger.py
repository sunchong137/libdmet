import sys
from datetime import datetime

"""
Logger
"""

Level = dict(zip("FATAL ERR SECTION RESULT WARNING INFO DEBUG0 DEBUG1 DEBUG2".split(), range(8)))

stdout = sys.stdout
verbose = "INFO"
clock = True

def __verbose():
    return Level[verbose]

def fatal(msg, *args):
    if __verbose() >= Level['FATAL']:
        __clock()
        flush("  FATAL ", msg, *args, indent = clock * 19)

def fassert(cond, msg, *args):
    if not cond:
        fatal(msg, *args)
        raise Exception

def error(msg, *args):
    if __verbose() >= Level['ERR']:
        __clock()
        flush("  ERROR ", msg, *args, indent = clock * 19)

def eassert(cond, msg, *args):
    if not cond:
        error(msg, *args)
        raise Exception

def section(msg, *args):
    if __verbose() >= Level['SECTION']:
        __clock()
        flush("####### ", msg, *args, indent = clock * 19)

def result(msg, *args):
    if __verbose() >= Level['RESULT']:
        __clock()
        flush("******* ", msg, *args, indent = clock * 19)

def warning(msg, *args):
    if __verbose() >= Level['WARNING']:
        __clock()
        flush("WARNING ", msg, *args, indent = clock * 19)

def check(cond, msg, *args):
    if not cond:
        warning(msg, *args)

def info(msg, *args):
    if __verbose() >= Level['INFO']:
        __clock()
        flush("   INFO ", msg, *args, indent = clock * 19)

def debug(level, msg, *args):
    if __verbose() >= Level["DEBUG0"] + level:
        __clock()
        flush("  DEBUG " + "  " * level, msg, *args, indent = clock * 19)

def time():
    stdout.write(datetime.now().strftime("%y %b %d %H:%M:%S") + "\n")
    stdout.flush()

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
        stdout.write(datetime.now().strftime("%y %b %d %H:%M:%S") + " ")

section("libDMET ---- A Period DMET Code\n\tby. Bo-Xiao Zheng\n\t<boxiao.zheng@gmail.com>")

def test():
    result("Logger Levels: %s", Level)
    warning("Logger Levels: %s", Level)
    info("Logger Levels: %s", Level)
    debug(0, "Logger Levels: %s", Level)
    debug(1, "Logger Levels: %s", Level)
    debug(2, "Logger Levels: %s", Level)
    error("Logger Levels: %s", Level)
    fatal("Logger Levels: %s", Level)


if __name__ == "__main__":
    test()
