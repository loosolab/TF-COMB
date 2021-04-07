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

	def error(self, msg):	#level 0
		return(self.log(ERROR, msg))

	def info(self, msg):
		return(self.log(INFO, msg))

	#def debug(self, msg):	

"""
	def _set_log_file(settings):
		file = settings.logfile
		name = settings.logpath
		root = settings._root_logger
		h = logging.StreamHandler(file) if name is None else logging.FileHandler(name)
		h.setFormatter(_LogFormatter())
		h.setLevel(self.level)
		if len(root.handlers) == 1:
			root.removeHandler(root.handlers[0])
		elif len(root.handlers) > 1:
			raise RuntimeError('Scanpy’s root logger somehow got more than one handler')
	root.addHandler(h)
"""
"""
def _set_log_level(settings, level: int):

	root = settings._root_logger
	root.setLevel(level)
	h, = root.handlers  # may only be 1
	h.setLevel(level)
"""

#Formatter for log
class _LogFormatter(logging.Formatter):

	def __init__(self, fmt='{levelname}: {message}', datefmt='%Y-%m-%d %H:%M', style='{'):
		super().__init__(fmt, datefmt, style)

#Functions imported
def warning(msg, *, time=None, deep=None, extra=None):
	return _TFcombLogger.warning(msg, time=time, deep=deep, extra=extra)

def info(msg, *, time=None, deep=None, extra=None):
	return _TFcombLogger.info(msg, time=time, deep=deep, extra=extra)
