import logging
from logging import ERROR, WARNING, INFO, DEBUG
import sys


#Create additional level
SPAM = DEBUG - 1
logging.addLevelName(SPAM, 'SPAM')

verbosity_to_level = {0: ERROR, #essentially silent
					  1: WARNING,
					  2: INFO,
					  3: DEBUG,
					  4: SPAM #extreme spam debugging
					  } 
					  

class TFcombLogger(logging.RootLogger):

	def __init__(self, verbosity):

		if verbosity not in verbosity_to_level.keys():
			raise ValueError("Verbosity level {0} is not valid. Please choose one of: 0 (only errors), 1 (warnings), 2 (info - default), 3 (debug), 4 (spam debug).".format(verbosity))

		self.level = verbosity_to_level[verbosity]
		super().__init__(self.level)

		self.formatter = _LogFormatter()
		self._setup()

	def _setup(self):

		con = logging.StreamHandler(sys.stdout)		#console output
		con.setLevel(self.level)
		con.setFormatter(self.formatter)
		self.addHandler(con)	

	def log(self, level, msg):
		super().log(level, msg)

	def spam(self, msg):
		return(self.log(SPAM, msg))

#Formatter for log
class _LogFormatter(logging.Formatter):

	def __init__(self, fmt='{levelname}: {message}', datefmt='%Y-%m-%d %H:%M', style='{'):
		super().__init__(fmt, datefmt, style)

#Functions imported
def warning(msg, *, time=None, deep=None, extra=None):
	return _TFcombLogger.warning(msg, time=time, deep=deep, extra=extra)

def info(msg, *, time=None, deep=None, extra=None):
	return _TFcombLogger.info(msg, time=time, deep=deep, extra=extra)
