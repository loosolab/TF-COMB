import logging
from logging import ERROR, INFO, DEBUG
import sys

#Create additional level
SPAM = DEBUG - 1
logging.addLevelName(SPAM, 'SPAM')

verbosity_to_level = {0: ERROR, #essentially silent
					  1: INFO,
					  2: DEBUG,
					  3: SPAM #extreme spam debugging
					  } 
					  
class InputError(Exception):
	""" Raises an InputError exception without writing traceback """

	def _render_traceback_(self):
		etype, msg, tb = sys.exc_info()
		sys.stderr.write("{0}: {1}".format(etype.__name__, msg))

class TFcombLogger(logging.RootLogger):

	def __init__(self, verbosity):

		if verbosity not in verbosity_to_level.keys():
			raise InputError("Verbosity level {0} is not valid. Please choose one of: 0 (only errors), 1 (info - default), 2 (debug), 3 (spam debug).".format(verbosity))

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

