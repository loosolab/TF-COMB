import sys
import os
import pandas as pd
import numpy as np
import copy
from copy import deepcopy
import time
import datetime
import scipy.stats
from scipy.signal import find_peaks
import random
import string
import multiprocessing as mp
from kneed import DataGenerator, KneeLocator
import statsmodels.stats.multitest
import importlib

import pysam
import tfcomb
from tfcomb.logging import TFcombLogger, InputError
from tfcomb.counting import count_co_occurrence
from tobias.utils.regions import OneRegion, RegionList
from tobias.utils.motifs import MotifList
from tobias.utils.signals import fast_rolling_math
import pathlib
import pyBigWig

#----------------- Minimal TFBS class based on the TOBIAS 'OneRegion' class -----------------#

class OneTFBS():
	""" Collects location information about one single TFBS """

	def __init__(self, **kwargs):

		#Initialize attributes
		for att in ["chrom", "start", "end", "name", "score", "strand"]:
			setattr(self, att, None)

		#Overwrite attributes with kwargs
		for att, value in kwargs.items():
			setattr(self, att, value)

		self.get_width = OneRegion.get_width
	
	def __str__(self):
		elements = [self.chrom, self.start, self.end, self.name, self.score, self.strand]
		elements = [str(element) for element in elements if element is not None]
		return("\t".join(elements))

	def __repr__(self):
		return(self.__str__())

	def from_oneregion(self, oneregion):
		self.chrom = oneregion.chrom
		self.start = oneregion.start
		self.end = oneregion.end
		self.name = oneregion.name
		self.score = oneregion.score
		self.strand = oneregion.strand

		return(self)

class TFBSPair():
	""" Collects information about a co-occurring pair of TFBS """

	def __init__(self, TFBS1, TFBS2, distance, directional=False):

		self.site1 = TFBS1 #OneTFBS object
		self.site2 = TFBS2 #OneTFBS object
		self.distance = distance

		#Calculate orientation scenario
		if directional == True:

			self.orientation = "TF1-TF2"
			self.orientation = "TF2-TF1"
			self.orientation = "divergent"
			self.orientation = "convergent"

		else:

			if self.site1.strand == self.site2.strand:
				self.orientation = "same"
			else:
				self.orientation = "opposite"

	def __str__(self):
		TFBS1 = ",".join([str(getattr(self.site1, col)) for col in ["chrom", "start", "end", "name", "score", "strand"]])
		TFBS2 = ",".join([str(getattr(self.site2, col)) for col in ["chrom", "start", "end", "name", "score", "strand"]])

		s = f"<TFBSPair | TFBS1: ({TFBS1}) | TFBS2: ({TFBS2}) | distance: {self.distance} | orientation: {self.orientation} >"
		return(s)

	def __repr__(self):
		return(self.__str__())

#------------------------------ Notebook / script exceptions -----------------------------#

def _is_notebook():
	""" Utility to check if function is being run from a notebook or a script """
	try:
		ipython_shell = get_ipython()
		return(True)
	except NameError:
		return(False)

class InputError(Exception):
	""" Raises an InputError exception without writing traceback """

	def _render_traceback_(self):
		etype, msg, tb = sys.exc_info()
		sys.stderr.write("{0}: {1}".format(etype.__name__, msg))

class StopExecution(Exception):
	""" Stop execution of a notebook cell with error message"""

	def _render_traceback_(self):
		etype, msg, _ = sys.exc_info()
		sys.stderr.write("{1}".format(etype.__name__, msg))
		#sys.stderr.write(f"{msg}")

def check_graphtool():
	""" Utility to check if 'graph-tool' is installed on path. Raises an exception (if notebook) or exits (if script) if the module is not installed. """

	error = 0
	try:
		import graph_tool.all
	except ModuleNotFoundError:
		error = 1
	except: 
		raise #unexpected error loading module
	
	#Write out error if module was not found
	if error == 1:
		s = "ERROR: Could not find the 'graph-tool' module on path. This module is needed for some of the TFCOMB network analysis functions. "
		s += "Please visit 'https://graph-tool.skewed.de/' for information about installation."

		if _is_notebook():
			raise StopExecution(s) from None
		else:
			sys.exit(s)
	
	return(True)

def check_module(module):
	""" Check if <module> can be imported without error """

	error = 0
	try:
		importlib.import_module(module)
	except ModuleNotFoundError:
		error = 1
	except: 
		raise #unexpected error loading module
	
	#Write out error if module was not found
	if error == 1:
		s = f"ERROR: Could not find the '{module}' module on path. This module is needed for this functionality. Please install this package to proceed."

		if _is_notebook():
			raise StopExecution(s) from None
		else:
			sys.exit(s)
	
	return(True)

#--------------------------------- File/type checks ---------------------------------#

def check_columns(df, columns):
	""" Utility to check whether columns are found within a pandas dataframe.
	
	Parameters 
	------------
	df : pandas.DataFrame
		A pandas dataframe to check.
	columns : list
		A list of column names to check for within 'df'.

	Raises
	--------
	InputError
		If any of the columns are not in 'df'.
	"""
	
	df_columns = df.columns

	not_found = []
	for column in columns:
		if column is not None:
			if column not in df_columns:
				not_found.append(column)
	
	if len(not_found) > 0:
		error_str = "Columns '{0}' are not found in dataframe. Available columns are: {1}".format(not_found, df_columns)
		raise InputError(error_str)
		
def check_dir(dir_path, create=True):
	""" Check if a dir is writeable.
	
	Parameters
	------------
	dir_path : str
		A path to a directory.
	
	Raises
	--------
	InputError
		If dir_path is not writeable.
	"""
	#Check if dir already exists
	if dir_path is not None: #don't check path given as None; assume that this is taken care of elsewhere
		if os.path.exists(dir_path):
			if not os.path.isdir(dir_path): # is it a file or a dir?
				raise InputError("Path '{0}' is not a directory".format(dir_path))

		#check writeability of parent dir
		else:
			if create:
				pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
			

def check_writeability(file_path):
	""" Check if a file is writeable.
	
	Parameters
	------------
	file_path : str
		A path to a file.
	
	Raises
	--------
	InputError
		If file_path is not writeable.
	"""

	check_type(file_path, str)

	#Check if file already exists
	error_str = None
	if os.path.exists(file_path):
		if not os.path.isfile(file_path): # is it a file or a dir?
			error_str = "Path '{0}' is not a file".format(file_path)

	#check writeability of parent dir
	else:
		pdir = os.path.dirname(file_path)
		pdir = "." if pdir == "" else pdir #if file_path is in current folder, pdir is empty string
		if os.access(pdir, os.W_OK) == False:
			error_str = "Parent directory '{0}' is not writeable".format(pdir)

	#If any errors were found
	if error_str is not None:
		raise InputError(error_str)


def check_type(obj, allowed, name=None):
	"""
	Check whether given object is within a list of allowed types.

	Parameters
	------------
	obj : object
		Object to check type on
	allowed : type or list of types
		A type or a list of object types to be allowed
	name : str, optional
		Name of object to be written in error. Default: None (the input is referred to as 'object')

	Raises
	--------
	InputError
		If object type is not within types.
	"""

	#Convert allowed to list
	if not isinstance(allowed, list):
		allowed = [allowed]

	#Check if any of the types fit
	flag = 0
	for t in allowed:
		if isinstance(obj, t):
			flag = 1

	#Raise error if none of the types fit
	if flag == 0:
		name = "object" if name is None else f'\'{name}\''
		raise InputError("The {0} given has type '{1}', but must be one of: {2}".format(name, type(obj), allowed))

def check_string(astring, allowed, name=None):
	""" 
	Check whether given string is within a list of allowed strings.
	
	Parameters
	------------
	astring : str
		A string to check.
	allowed : str or list of strings
		A string or list of allowed strings to check against 'astring'.
	name : str, optional
		The name of the string to be written in error. Default: None (the value is referred to as 'string').

	Raises
	--------
	InputError
		If 'astring' is not in 'allowed'.
	"""

	#Convert allowed to list
	if not isinstance(allowed, list):
		allowed = [allowed]
	
	#Check if astring is within allowed
	if astring not in allowed:
		name = "string" if name is None else f'\'{name}\''
		raise InputError("The {0} given ({1}) is not valid - it must be one of: {2}".format(name, astring, allowed))

def check_value(value, vmin=-np.inf, vmax=np.inf, integer=False, name=None):
	"""
	Check whether given 'value' is a valid value (or integer) and if it is within the bounds of vmin/vmax.

	Parameters
	-------------
	value : int or float
		The value to check.
	vmin : int or float, optional
		Minimum the value is allowed to be. Default: -infinity (no bound)
	vmax : int or float
		Maxmum the value is allowed to be. Default: +infinity (no bound)
	integer : bool, optional
		Whether value must be an integer. Default: False (value can be float)
	name : str, optional
		The name of the value to be written in error. Default: None (the value is referred to as 'value').

	Raises
	--------
	InputError
		If 'value' is not a valid value as given by parameters.
	"""

	if vmin > vmax:
		raise InputError("vmin must be smaller than vmax")

	error_msg = None
	if integer == True:		
		if not isinstance(value, int):
			error_msg = "The {0} given ({1}) is not an integer, but integer is set to True.".format(name, value)
	else:
		#check if value is any value
		try:
			_ = int(value)
		except:
			error_msg = "The {0} given ({1}) is not a valid number".format(name, value)

	#If value is a number, check if it is within bounds
	if error_msg is None:
		if not ((value >= vmin) & (value <= vmax)):
			error_msg = "The {0} given ({1}) is not within the bounds of [{2};{3}]".format(name, value, vmin, vmax)
	
	#Finally, raise error if necessary:
	if error_msg is not None:
		raise InputError(error_msg)

def random_string(l=8):
	""" Get a random string of length l """
	s = ''.join(random.choice(string.ascii_uppercase) for _ in range(l))
	return(s)

#--------------------------------- Multiprocessing ---------------------------------#

class Progress():

	def __init__(self, n_total, n_print, logger):
		"""
		Utility class to monitor progress of a list of tasks.
		
		Parameters
		-----------
		n_total : int
			Number of total jobs
		n_print : int
			Number of times to write progress
		logger : logger instance
			The logger to use for writing progress  
		"""
		
		self.n_total = n_total
		self.n_print = n_print
		self.logger = logger        

		#At what number of tasks should the updates be written?
		n_step = int(n_total / (n_print))
		self.progress_steps = [n_step*(i+1) for i in range(n_print)]
		
		self.next = self.progress_steps[0] #first limit in progress_steps to write


	def write_progress(self, n_done):
		""" Log the progress of the current tasks.
		
		Parameters
		-----------
		n_done : int
			Number of tasks done (of n_total tasks)
		"""
		
		if n_done >= self.next:
			
			self.logger.info("Progress: {0:.0f}%".format(n_done/self.n_total*100.0))
			
			#in case more than one progress step was jumped
			remaining_steps = [step for step in self.progress_steps if n_done < step] + [np.inf]
			
			self.next = remaining_steps[0]    #this is the next idx to write (or np.inf if end of list was reached)

def log_progress(jobs, logger, n=10):
	""" 
	Log progress of jobs within job list.

	Parameters
	------------
	jobs : list
		List of multiprocessing jobs to write progress for.
	logger : logger instance
		A logger to use for writing out progress.
	n : int, optional
		Maximum number of progress statements to show. Default: 10. 
	"""

	#Setup progress obj
	n_tasks = len(jobs)
	p = Progress(n_tasks, n, logger)

	n_done = sum([task.ready() for task in jobs])
	while n_done != n_tasks:
		p.write_progress(n_done)
		time.sleep(0.1)
		n_done = sum([task.ready() for task in jobs]) #recalculate number of finished jobs

	logger.info("Finished!")

	return(0) 	#doesn't return until the while loop exits


#--------------------------------------- Motif / TFBS scanning and processing ---------------------------------------#

def prepare_motifs(motifs_file, motif_pvalue=0.0001, motif_naming="name"):
	""" Read motifs from motifs_file and set threshold/name. """

	#Read and prepare motifs
	motifs_obj = MotifList().from_file(motifs_file)

	_ = [motif.get_threshold(motif_pvalue) for motif in motifs_obj]
	_ = [motif.set_prefix(motif_naming) for motif in motifs_obj] #using naming from args

	return(motifs_obj)

def open_genome(genome_f):	
	""" Opens an internal genome object for fetching sequences.

	Parameters
	------------
	genome_f : str
		The path to a fasta file.
	
	Returns
	---------
	pysam.FastaFile
	"""

	genome_obj = pysam.FastaFile(genome_f)
	return(genome_obj)

def open_bigwig(bigwig_f):
	"""
	Parameters
	------------
	bigwig_f : str
		The path to a bigwig file.	
	
	"""

	pybw_obj = pyBigWig.open(bigwig_f)

	return(pybw_obj)

def check_boundaries(regions, genome):
	""" Utility to check whether regions are within the boundaries of genome.
	
	Parameters
	-----------
	regions : tobias.utils.regions.RegionList 
		A RegionList() object containing regions to check.
	genome : pysam.FastaFile
		An object (e.g. from open_genome()) to use as reference. 
	
	Raises
	-------
	InputError
		If a region is not available within genome
	"""

	chromosomes = genome.references
	lengths = genome.lengths
	genome_bounds = dict(zip(chromosomes, lengths))

	for region in regions:
		if region.chrom not in chromosomes:
			raise InputError("Region '{0} {1} {2} {3}' is not present in the given genome. Available chromosomes are: {4}.".format(region.chrom, region.start, region.end, region.name, chromosomes))
		else:
			if region.start < 0 or region.end > genome_bounds[region.chrom]:
				raise InputError("Region '{0} {1} {2} {3}' is out of bounds in the given genome. The length of the chromosome is: {4}".format(region.chrom, region.start, region.end, region.name, genome_bounds[region.chrom]))


def unique_region_names(regions):
	""" 
	Get a list of unique region names within regions. 

	Parameters
	-----------
	regions : tobias.utils.regions.RegionList 
		A RegionList() object containing regions with .name attributes.

	Returns
	--------
	list
		The list of sorted names from regions.
	"""

	names_dict = {r.name: True for r in regions}
	names = sorted(list(names_dict.keys()))

	return(names)

def calculate_TFBS(regions, motifs, genome, resolve="merge"):
	"""
	Multiprocessing-safe function to scan for motif occurrences

	Parameters
	----------
	genome : str or 
		If string , genome will be opened 
	regions : RegionList()
		A RegionList() object of regions 
	resolve : str
		How to resolve overlapping sites from the same TF. Must be one of "off", "highest_score" or "merge". If "highest_score", the highest scoring overlapping site is kept.
		If "merge", the sites are merged, keeping the information of the first site. If "off", overlapping TFBS are kept. Default: "merge".

	Returns
	----------
	List of TFBS within regions

	"""

	check_string(resolve, ["merge", "highest_score", "off"], "resolve")

	#open the genome given
	if isinstance(genome, str):
		genome_obj = open_genome(genome)
	else:
		genome_obj = genome

	TFBS_list = RegionList()
	for region in regions:
		seq = genome_obj.fetch(region.chrom, region.start, region.end)
		region_TFBS = motifs.scan_sequence(seq, region)

		#Convert RegionLists to TFBS class
		region_TFBS = RegionList([OneTFBS().from_oneregion(region) for region in region_TFBS])
		region_TFBS.loc_sort()

		TFBS_list += region_TFBS

	#Sort all sites
	TFBS_list.loc_sort()

	#Resolve overlapping
	if resolve != "off":
		TFBS_list = resolve_overlaps(TFBS_list, how=resolve)

	if isinstance(genome, str):
		genome_obj.close()

	return(TFBS_list)

def resolve_overlaps(sites, how="merge", per_name=True):
	""" 
	Resolve overlapping sites within a list of genomic regions.

	Parameters
	------------
	sites : RegionList
		A list of TFBS/regions with .chrom, .start, .end and .name information.
	how : str
		How to resolve the overlapping site. Must be one of "highest_score", "merge". If "highest_score", the highest scoring overlapping site is kept.
		If "merge", the sites are merged, keeping the information of the first site. Default: "merge".
	per_name : bool
		Whether to resolve overlapping only per name or across all sites. If 'True' overlaps are only resolved if the name of the sites are equal. 
		If 'False', overlaps are resolved across all sites. Default: True.
	"""
	
	check_string(how, ["highest_score", "merge"], "how")

	#Create a copy of sites to ensure that original sites are not changed
	sites = copy.copy(sites)
	
	n_sites = len(sites)
	tracking = {} # dictionary for tracking positions of TFBS per name (or across all)
	
	for current_site_i in range(n_sites):
		
		current_site = sites[current_site_i]
		site_name = current_site.name if per_name == True else "." #control which site to fetch as 'previous'
		
		if site_name in tracking: #if not in tracking, site is the first site of this name
			
			#previous_site = tracking[site_name]["site"]
			previous_i = tracking[site_name]
			previous_site = sites[previous_i]

			if (current_site.chrom == previous_site.chrom) and (current_site.start < previous_site.end): #overlapping
								
				#How to deal with overlap:
				if how == "highest_score":
					
					if current_site.score >= previous_site.score: #keep current site
						sites[previous_i] = None
						tracking[site_name] = current_site_i #new tracking
						
					else: #keep previous site
						sites[current_site_i] = None
						#tracking stays the same
						
				elif how == "merge":
					
					merged_end = max([previous_site.end, current_site.end])
					
					#merge site into the previous; keep previous score/strand
					merged = OneTFBS(**{"chrom": current_site.chrom, 
									  "start": previous_site.start, 
									  "end": merged_end, 
									  "name": previous_site.name, 
									  "score": previous_site.score,
									  "strand": previous_site.strand})
					
					sites[previous_i] = merged
					sites[current_site_i] = None
					#tracking i stays the same

					#tracking[site_name] = previous_i, "site": merged} , but site is updated to merged
					
			else: #no overlaps with previous; save this site to tracking
				tracking[site_name] = current_site_i
				
		else: #Save first site to tracking
			tracking[site_name] = current_site_i
	
	resolved = [site for site in sites if site is not None]
	
	return(resolved)


#----------------------------- Analysis on pairs of TFBS ------------------------#

def get_pair_locations(sites, TF1, TF2, TF1_strand = None,
										   TF2_strand = None,
										   min_distance = 0, 
										   max_distance = 100, 
										   max_overlap = 0,
										   directional = False,
										   anchor = "inner"):
	""" Get genomic locations of a particular TF pair.
	
	Parameters
	----------
	sites : RegionList()
		A list of TFBS regions.
	TF1 : str 
		Name of TF1 in pair.
	TF2 : str 
		Name of TF2 in pair.
	TF1_strand : str, optional
		Strand of TF1 in pair. Default: None (strand is not taken into account).
	TF2_strand : str, optional
		Strand of TF2 in pair. Default: None (strand is not taken into account).
	min_distance : int, optional
		Minimum distance allowed between two TFBS. Default: 0
	max_distance : int, optional
		Maximum distance allowed between two TFBS. Default: 100
	max_overlap : float between 0-1, optional
		Controls how much overlap is allowed for individual sites. A value of 0 indicates that overlapping TFBS will not be saved as co-occurring. 
		Float values between 0-1 indicate the fraction of overlap allowed (the overlap is always calculated as a fraction of the smallest TFBS). A value of 1 allows all overlaps. Default: 0 (no overlap allowed).
	directional : bool, optional
		Decide if direction of found pairs should be taken into account, e.g. whether  "<---TF1---> <---TF2--->" is only counted as 
		TF1-TF2 (directional=True) or also as TF2-TF1 (directional=False). Default: False.
	anchor : str, optional
		The anchor to use for calculating distance. Must be one of ["inner", "outer", "center"]

	Returns
	-------
	List of TFBSPair objects
		Each entry in the list is a TFBSPair object, which contains .site1, .site2, .distance and .orientation variables

	See also
	---------
	count_within

	"""

	#Check input types
	check_string(anchor, ["inner", "outer", "center"], "anchor")

	#Subset sites to TF1/TF2 sites
	sites = [site for site in sites if site.name in [TF1, TF2]]

	locations = [] #empty list of regions

	TF1_tup = (TF1, TF1_strand)
	TF2_tup = (TF2, TF2_strand)
	n_sites = len(sites)

	#Find out which TF is queried
	if directional == True:
		TF1_to_check = [TF1_tup]
	else:
		TF1_to_check = [TF1_tup, TF2_tup]

	#Loop over all sites
	i = 0
	while i < n_sites: #i is 0-based index, so when i == n_sites, there are no more sites
		
		#Get current TF information
		TF1_chr, TF1_start, TF1_end, TF1_name, TF1_strand_i = sites[i].chrom, sites[i].start, sites[i].end, sites[i].name, sites[i].strand
		this_TF1_tup = (TF1_name, None) if TF1_tup[-1] == None else (TF1_name, TF1_strand_i)

		#Check whether TF is valid
		if this_TF1_tup in TF1_to_check:

			#Find possible associations with TF1 within window 
			finding_assoc = True
			j = 0
			while finding_assoc == True:
				
				#Next site relative to TF1
				j += 1
				if j+i >= n_sites - 1: #next site is beyond end of list, increment i
					i += 1
					finding_assoc = False #break out of finding_assoc

				else:	#There are still sites available

					#Fetch information on TF2-site
					TF2_chr, TF2_start, TF2_end, TF2_name, TF2_strand_i = sites[i+j].chrom, sites[i+j].start, sites[i+j].end, sites[i+j].name, sites[i+j].strand
					this_TF2_tup = (TF2_name, None) if TF2_tup[-1] == None else (TF2_name, TF2_strand_i)	
					
					#Find out whether this TF2 is TF1/TF2
					if this_TF1_tup == TF1_tup:
						to_check = TF2_tup
					elif this_TF1_tup == TF2_tup:
						to_check = TF1_tup

					#Check whether TF2 is either TF1/TF2
					if this_TF2_tup == to_check:
					
						#Calculate distance between the two sites based on anchor
						if anchor == "inner":
							distance = TF2_start - TF1_end #TF2_start - TF1_end will be negative if TF1 and TF2 are overlapping
							if distance < 0:
								distance = 0
						elif anchor == "outer":
							distance = TF2_end - TF1_start
						elif anchor == "center":
							TF1_mid = (TF1_start + TF1_end) / 2
							TF2_mid = (TF2_start + TF2_end) / 2
							distance = TF2_mid - TF1_mid

						#True if these TFBS co-occur within window
						if TF1_chr == TF2_chr and (distance <= max_distance):

							if distance >= min_distance:
							
								# check if they are overlapping more than the threshold
								valid_pair = 1
								if distance == 0:

									# Get the length of the shorter TF
									short_bp = min([TF1_end - TF1_start, TF2_end - TF2_start])

									#Calculate overlap between TF1/TF2
									overlap_bp = TF1_end - TF2_start #will be negative if no overlap is found
									if overlap_bp > short_bp: #overlap_bp can maximally be the size of the smaller TF (is larger when TF2 is completely within TF1)
										overlap_bp = short_bp
									
									#Invalid pair, overlap is higher than threshold
									if overlap_bp / (short_bp*1.0) > max_overlap: 
										valid_pair = 0

								#Save association
								if valid_pair == 1:

									#Save location
									reg1 = sites[i] 
									reg2 = sites[i+j]
									pair = TFBSPair(reg1, reg2, distance, directional=directional)
									locations.append(pair)
						elif TF1_chr != TF2_chr: 
							i += 1
							finding_assoc = False   #break out of finding_assoc-loop

						else: #This TF2 is on the same chromosome but more than max_distance away

							#Establish if all valid sites were found for TF1
							if anchor == "inner":

								#The next site is out of inner window range; increment to next i
								i += 1
								finding_assoc = False   #break out of finding_assoc-loop
							
							else: #If anchor is outer or center, there might still be valid pairs for future TF2's

								#Check if it will be possible to find valid pairs in next sites
								if TF2_start > TF1_start + max_distance:
									#no longer possible to find valid pairs for TF1; increment to next i
									i += 1
									finding_assoc = False   #break out of finding_assoc-loop
		
		else: #current TF1 is not TF1/TF2; go to next site
			i += 1

	return(locations)

def locations_to_bed(locations, outfile, fmt="bed"):
	""" 
	Write the locations of (TF1, TF2) pairs to a bed-file.
	
	Parameters
	------------
	locations : list
		The output of get_pair_locations().
	outfile : str
		The path which the pair locations should be written to.
	fmt : str, optional
		The format of the output file. Must be one of "bed" or "bedpe". If "bed", the TF1/TF2 sites will be written as one region spanning TF1.start-TF2.end. If "bedpe", the sites are written in BEDPE format. Default: "bed".
	"""
	
	tfcomb.utils.check_string(fmt, ["bed", "bedpe"], "fmt")

	#Open output file
	try:
		f = open(outfile, "w")
	except Exception as e:
		raise InputError("Error opening '{0}' for writing. Error message was: {1}".format(outfile, e))
	
	#Write locations to file in format 'fmt'
	if fmt == "bed":
		s = "\n".join(["\t".join([l.site1.chrom, str(l.site1.start), str(l.site2.end), l.site1.name + "-" + l.site2.name, str(l.distance), "."]) for l in locations]) + "\n"

	elif fmt == "bedpe":
		s = "\n".join(["\t".join([l.site1.chrom, str(l.site1.start), str(l.site1.end),
									l.site2.chrom, str(l.site2.start), str(l.site2.end), 
									l.site1.name + "-" + l.site2.name, str(l.distance), l.site1.strand, l.site2.strand]) for l in locations]) + "\n"
	f.write(s)
	f.close()


#--------------------------------- Background calculation ---------------------------------#

def shuffle_array(arr, seed=1):
	np.random.seed(seed)
	length = arr.shape[0]
	return(arr[np.random.permutation(length),:])

def shuffle_sites(sites, seed=1):
	""" Shuffle TFBS names to existing positions and updates lengths of the new positions.
	
	Parameters
	-----------
	sites : np.array
		An array of sites in shape (n_sites,4), where each row is a site and columns correspond to chromosome, start, end, name.
	
	Returns
	--------
	An array containing shuffled names with site lengths corresponding to original length of sites.
	"""
	
	#Establish lengths of regions
	lengths = sites[:,2] - sites[:,1]
	sites_plus = np.c_[sites, lengths]
	
	#Shuffle names (and corresponding lengths)
	sites_plus[:,-2:] = shuffle_array(sites_plus[:,-2:], seed)
	
	#Adjust coordinates to new length
	#new start = old start + old half length - new half length
	#new end = new start + new length
	sites_plus[:,1] = sites_plus[:,1] + ((sites_plus[:,2] - sites_plus[:,1])/2) - sites_plus[:,-1]/2 #new start
	sites_plus[:,2] = sites_plus[:,1] + sites_plus[:,-1] #new end
	
	#Remove length again
	sites_shuffled = sites_plus[:,:-1]
	
	return(sites_shuffled)

def calculate_background(sites, min_distance, 
								max_distance, 
								max_overlap,
								binary,
								anchor,
								n_TFs,
								directional,
								seed=1):
	""" 
	Wrapper to shuffle sites and count co-occurrence of the shuffled sites. 
	
	Parameters
	------------
	sites : np.array
		An array of sites in shape (n_sites,4), where each row is a site and columns correspond to chromosome, start, end, name.
	min_distance
	max_distance
	max_overlap
	binary
	anchor
	n_TFs
	directional
	seed
	"""
	
	#Shuffle sites
	s = datetime.datetime.now()
	shuffled = shuffle_sites(sites, seed=seed)
	e = datetime.datetime.now()
	#print("Shuffling: {0}".format(e-s))
	
	s = datetime.datetime.now()
	_, pair_counts = count_co_occurrence(shuffled, 
													min_distance,
													max_distance,
													max_overlap, 
													binary,
													anchor,
													n_TFs)
	e = datetime.datetime.now()
	#print("counting: {0}".format(e-s))
	pair_counts = tfcomb.utils.make_symmetric(pair_counts) if directional == False else pair_counts	#Deal with directionality
	
	return(pair_counts)


#--------------------------------- P-value calculation ---------------------------------#

def get_threshold(data, which="upper", percent=0.05, _n_max=10000, verbosity=0):
	"""
	Function to get upper/lower threshold(s) based on the distribution of data. The threshold is calculated as the probability of "percent" (upper=1-percent).
	
	Parameters
	------------
	data : list or array
		An array of data to find threshold on.
	which : str
		Which threshold to calculate. Can be one of "upper", "lower", "both". Default: "upper".
	percent : float between 0-1
		Controls how strict the threshold should be set in comparison to the distribution. Default: 0.05.
	
	Returns
	---------
	If which is one of "upper"/"lower", get_threshold returns a float. If "both", get_threshold returns a list of two float thresholds.
	"""
	
	distributions = [scipy.stats.norm, scipy.stats.lognorm, scipy.stats.laplace, 
					 scipy.stats.expon, scipy.stats.truncnorm, scipy.stats.truncexpon, scipy.stats.wald, scipy.stats.weibull_min]
	
	logger = tfcomb.logging.TFcombLogger(verbosity)

	#Check input parameters
	check_string(which, ["upper", "lower", "both"], "which")
	check_value(percent, vmin=0, vmax=1, name="percent")

	#Subset data to _n_max:
	if len(data) > _n_max:
		np.random.seed(0)
		data = np.random.choice(data, size=_n_max, replace=False)
	
	data_finite = np.array(data)[~np.isinf(data)]

	#Fit data to each distribution
	distribution_dict = {}
	for distribution in distributions:
		logger.debug("Fitting data to '{0}'".format(distribution))
		params = distribution.fit(data_finite)

		#Test fit using negative loglikelihood function
		mle = distribution.nnlf(params, data_finite)

		#Save info on distribution fit    
		distribution_dict[distribution.name] = {"distribution": distribution,
												"params": params, 
												"mle": mle}

	#Get best distribution
	best_fit_name = sorted(distribution_dict, key=lambda x: distribution_dict[x]["mle"])[0]
	parameters = distribution_dict[best_fit_name]["params"]
	best_distribution = distribution_dict[best_fit_name]["distribution"]

	#Get threshold
	thresholds = best_distribution(*parameters).ppf([percent, 1-percent])
	
	if which == "upper":
		final = thresholds[-1]
	elif which == "lower":
		final = thresholds[0]
	elif which == "both":
		final = tuple(thresholds)
		
	#Plot fit and threshold
	#plt.hist(data, bins=20, density=True)
	#xmin = np.min(data)
	#xmax = np.max(data)
	#x = np.linspace(xmin, xmax, 100)
	#plt.plot(x, best_distribution(*params).pdf(x), lw=5, alpha=0.6, label=best_distribution.name)
	#plt.legend()
	
	return(final)

def tfcomb_pvalue(table, measure="cosine", alternative="greater", threads = 1, logger=None):
	"""
	Calculates the p-value of each TF1-TF2 pair for the measure given.

	Parameters
	------------
	table : pd.DataFrame
		The table from '.rules' of DiffObj or DiffCombObj.
	measure : str, optional
		The measure to calculate pvalue for. Default: "cosine".
	alternative : str, optional
		One of: 'two-sided', 'greater', 'less'. Default: "greater".
	threads : int, optional
		Number of threads to use for multiprocessing. Default: 1.
	logger : logger
		A logger to use for logging progress.
	"""
	
	if logger == None:
		logger = TFcombLogger(0) #silent logger

	#Check input types
	check_type(table, [pd.DataFrame], "table")
	check_type(measure, [str], "measure")
	check_type(alternative, [str], "alternative")
	check_type(threads, [int], "threads")

	#Create pivot-table from rules table
	pivot_table = pd.pivot(table, index="TF1", columns="TF2", values=measure)

	#Convert pivot table index/columns to integers
	TF_lists = {"TF1": pivot_table.index.tolist(),
				"TF2": pivot_table.columns.tolist()}
	pivot_table.index = [i for (i, _) in enumerate(TF_lists["TF1"])]
	pivot_table.columns = [i for (i, _) in enumerate(TF_lists["TF2"])]

	#Convert to matrix
	matrix = pivot_table.to_numpy()
	matrix = np.nan_to_num(matrix) #Fill NA with 0's
	
	##### Calculate background distribution for all TFs in TF1/TF2 #####
	bg_dist_dict = {"TF1": {}, "TF2": {}}

	for TF_number in bg_dist_dict: #first or second, e.g. TF1/TF2
		for i, _ in enumerate(TF_lists[TF_number]):
			
			if TF_number == "TF1":
				values = matrix[i,:]
			elif TF_number == "TF2":
				values = matrix[:,i]

			#Calculate information about values
			bg_dist_dict[TF_number][i] = {"n": len(values),
										"mu": np.mean(values),
										"std": np.std(values)
										}

	#Convert table entries to integers -> tuples
	tuples = table[["TF1", "TF2", measure]].to_records(index=False) #tuples of (TF1, TF2, value)
	TF1_idx = {TF: i for i, TF in enumerate(TF_lists["TF1"])}
	TF2_idx = {TF: i for i, TF in enumerate(TF_lists["TF2"])}
	tuples = [(TF1_idx[TF1], TF2_idx[TF2], value) for (TF1, TF2, value) in tuples]

	#Split tuples into individual tasks
	n_jobs = 100
	n_tuples = len(tuples)
	per_chunk = int(np.ceil(n_tuples/float(n_jobs)))
	tuples_chunks = [tuples[i:i+per_chunk] for i in range(0, n_tuples, per_chunk)]

	### Calculate pvalues with/without multiprocessing
	
	if threads == 1:
		pvalues = []
		p = Progress(n_jobs, 10, logger) #Setup progress object

		for i, chunk in enumerate(tuples_chunks):
			results = _pvalues_for_chunks(chunk, bg_dist_dict, alternative)
			pvalues.extend(results)
			p.write_progress(i)
		logger.info("Finished!")

	else:
		#start multiprocessing
		pool = mp.pool(threads)

		jobs = []
		for chunk in tuples_chunks:
			job = pool.apply_async(_pvalues_for_chunks, args=(chunk, bg_dist_dict, alternative, ))
			jobs.append(job)
		pool.close() 	#done sending jobs to pool

		log_progress(jobs)
		pvalues = [job.get() for job in jobs]
		pool.join()

		#Flatten results from individual jobs
		pvalues = sum(pvalues, [])


	##### Process pvalues #####

	#Add p-values to table
	col = measure + "_pvalue"
	table[col] = pvalues

	#no return, table is changed in place

def _pvalues_for_chunks(TF_int_combinations, bg_dist_dict, alternative="greater"):
	""" Wrapper of _calculate_pvalue for chunks of TF_int_combinations """

	pvalues = [""]*len(TF_int_combinations) #initialize list of p-values
	for i, (TF1_i, TF2_i, obs) in enumerate(TF_int_combinations):

		#Remove value from background distributions
		mu1, std1, n1 = remove_val_from_dist(obs, **bg_dist_dict["TF1"][TF1_i])
		mu2, std2, n2 = remove_val_from_dist(obs, **bg_dist_dict["TF2"][TF2_i])

		#Merge the two distributions
		mu = (n1 * mu1 + n2 * mu2)/(n1 + n2)
		std = np.sqrt((n1 * (std1**2 + (mu1-mu)**2) + n2*(std2**2 + (mu2-mu)**2))/(n1 + n2))

		#Calculate zscore->pvalue of observed value
		pvalues[i] = _calculate_pvalue(mu, std, obs, alternative=alternative) 

	return(pvalues)

def _calculate_pvalue(mu, std, obs, alternative="greater"):
	""" Calculate p-value of seeing value within the normal distribution with mu/std parameters"""
	
	if std == 0: #not possible to calculate p-value using zscore
		p = 0

	else:
		z = (obs - mu)/std
		p_oneside = scipy.stats.norm.sf(np.abs(z)) #one-sided pvalue

		#Calculate pvalue based on alternative hypothesis
		if alternative == "two-sided":
			p = p_oneside * 2

		elif alternative == "greater":
			if z > 0: #observed is larger than mean
				p = p_oneside
			else:
				p = 1.0 - p_oneside
		
		elif alternative == "less":
			if z < 0: #observed is smaller than mean
				p = p_oneside
			else:
				p = 1.0 - p_oneside

	return(p)


def remove_val_from_dist(value, mu, std, n):
	""" Remove a value from distribution 
	
	Parameters
	-----------
	value : float
		The value to remove from distribution. 
	mu : float
		Mean of the distribution.
	std : float
		Standard deviation of the distribution.
	n : int
		The number of elements in the distribution (including value).
	
	Returns
	-----------
		A tuple of (mean, std, n) for the new distribution
	"""

	zero_tol = 10**-10 #tolerance for zero due to decimal point rounding errors

	#Calculate new mean
	bg_mean = (mu*n - value)/(n-1)

	#Calculate new std
	if bg_mean < zero_tol: #No other values left
		bg_std = 0
	else:
		var = std**2
		bg_var = (var*(n-1) - (value - bg_mean) * (value - mu))/(n-2)
		bg_std = np.sqrt(bg_var)

	bg_n = n - 1

	return((bg_mean, bg_std, bg_n))


#--------------------------------- Working with TF-COMB objects ---------------------------------#

def is_symmetric(matrix):
	""" Check if a matrix is symmetric around the diagonal """
	b = np.allclose(matrix, matrix.T, equal_nan=True)
	return(b)

def make_symmetric(matrix):
	"""
	Make a numpy matrix matrix symmetric by merging x-y and y-x
	"""
	matrix_T = matrix.T 
	symmetric = matrix + matrix_T

	#don't add up diagonal indices
	di = np.diag_indices(symmetric.shape[0])
	symmetric[di] = matrix_T[di]

	return(symmetric)


def set_contrast(contrast, available_contrasts):
	""" Utility function for the plotting functions of tfcomb.objects.DiffCombObj """

	#Setup contrast to use
	if contrast == None:
		contrast = available_contrasts[0]

	else:
		#Check if contrast is tuple
		if contrast not in available_contrasts:
			raise ValueError("Contrast {0} is not valid (available contrasts are {1})".format(contrast, available_contrasts))

	return(contrast)

# ------------------------- chunk operations ---------------------------------------- #
def linress_chunks(pairs, dist_counts, distances):
	''' Helper function to process linear regression for chunks 
		
		Parameters
		-----------
		pairs: list<tuple>
			   A list of tuple with TF names (e.g. ("NFYA", "NFYB"))
		dist_counts: pd.DataFrame
			   A (sub-)Dataframe with the distance counts for the pairs
		distances: list
			   A list of valid column names for the distances  
		
		Returns
		-----------
		results: list 
				A list with the results in form of a list [TF1, TF2, LinearRegressionObject]
	'''
	# make sure index is correct
	dist_counts = dist_counts.reset_index()
	dist_counts.index = dist_counts["TF1"] + "-" + dist_counts["TF2"]
	
	distance_cols = np.array([-1 if d == "neg" else d for d in distances]) #neg counts as -1

	#save results as list
	results = []
	for pair in pairs:

		# get count for specific pair
		ind = "-".join(pair)
		counts = dist_counts.loc[ind].loc[distance_cols].values #exclude TF1, TF2 columns
		counts = np.array(counts, dtype=float)
		
		# fit linear regression
		res = scipy.stats.linregress(distances, counts)

		# get TF1, TF2 names from pair
		tf1, tf2 = pair
		results.append([tf1, tf2, res])
	return results

def correct_chunks(pairs, dist_counts, distances, linres):
	""" Subtracts the estimated background from the Signal for a given pair. 
			
	Parameters
	-----------
	pairs: list<tuple>
		   A list of tuple with TF names (e.g. ("NFYA", "NFYB"))
	dist_counts: pd.DataFrame
		   A (sub-)Dataframe with the distance counts for the pairs
	distances: list
		   A list of valid column names for the distances  
		
	Returns
	-----------
	results: list 
			A list with the results in form of a list [TF1, TF2, LinearRegressionObject]
	"""

	# make sure index is correct
	dist_counts = dist_counts.reset_index()
	dist_counts.index = dist_counts["TF1"] + "-" + dist_counts["TF2"]

	distance_cols = np.array([-1 if d == "neg" else d for d in distances]) #neg counts as -1

	linres = linres.reset_index()
	linres.index = linres["TF1"] + "-" + linres["TF2"]

	linres_col = "Linear Regression"
	
	#save results as list
	results = []
	for pair in pairs:

		# get count for specific pair
		ind = "-".join(pair)
		counts = dist_counts.loc[ind].loc[distance_cols].values #exclude TF1, TF2 columns
		counts = np.array(counts, dtype=float)

		linres_pair = linres.loc[ind].loc[linres_col] #exclude TF1, TF2 columns
	
		# subtract background
		corrected = counts - (linres_pair.intercept + linres_pair.slope * np.array(distances))
		
		# get TF1, TF2 names from pair
		tf1, tf2 = pair
		corrected = [tf1, tf2] + corrected.tolist()

		results.append(corrected)
		
	return results

def analyze_signal_chunks(pairs, datasource, distances, stringency, prominence):
	""" After background correction is done (see ._correct_pair() or .correct_all()), the signal is analyzed for peaks, 
		indicating prefered binding distances. There can be more than one peak (more than one prefered binding distance) per 
		Signal. Peaks are called with scipy.signal.find_peaks().
		
		Parameters
		----------
		pairs : list<tuple(str,str)>
			TF names for which the preferred binding distance(s) should be found. e.g. ("NFYA","NFYB")
		datasource : pd.DataFrame 
			A (sub-)Dataframe with the (corrected) distance counts for the pairs
		distances: list
			A list of valid column names for the distances  
		stringency: number
			stringency the prominence threshold should be multiplied with.
		prominence: number or ndarray or sequence
			prominence parameter for peak calling (see scipy.signal.find_peaks() for detailed information)
			Default: 0

		Returns:
		----------
		list 
			list of found peaks in form [TF1, TF2, Distance, Peak Heights, Prominences, Prominence Threshold]
	"""

	# make sure index is correct
	datasource = datasource.reset_index()
	datasource.index = datasource["TF1"] + "-" + datasource["TF2"]

	# get data column
	distance_cols = np.array([-1 if d == "neg" else d for d in distances]) #neg counts as -1

	results = []
	for pair in pairs:
		# get pair
		tf1, tf2 = pair
		ind = "-".join(pair)

		# signal.find_peaks() will not find peaks on first and last position without having 
		# an other number left and right. 
		signal = datasource.loc[ind].loc[distance_cols].values
		x = [0] + list(signal) + [0]

		# determine prominence
		if prominence =="zscore":
			prom = 1
		elif prominence =="median":
			prom = datasource.loc[ind].loc["median"].values
		else:
			prom = prominence
		
		# calc threshold 
		threshold = prom * stringency

		#Find positions of peaks
		peaks_idx, properties = find_peaks(x, prominence=threshold, height=threshold)

		# subtract the position added above (first zero) 
		peaks_idx = peaks_idx - 1 

		#Get distances from columns
		peak_distances = [distance_cols[idx] for idx in peaks_idx]

		'''
		#Collect peaks for TF pair
		peak_info = pd.DataFrame().from_dict(properties)
		peak_info["Distance"] = peak_distances
		peak_info["TF1"] = tf1
		peak_info["TF2"] = tf2

		#Add additional peak-information per pair
		peak_info["Threshold"] = threshold
		
		#Format order of columns
		columns = ["TF1", "TF2", "Distance", "peak_heights", "prominences", "Threshold"]
		peak_info = peak_info[columns]
		peak_info.rename(columns= { "peak_heights":"Peak Heights",
									"prominences": "Prominences"}, inplace=True)
		results.append(peak_info)
		'''
		n_peaks = len(peak_distances)
		# insert tf1,tf2 names number of peaks times
		properties["TF1"] = [tf1]*n_peaks
		properties["TF2"] = [tf2]*n_peaks

		properties["Distance"] = peak_distances
		properties["Threshold"] = threshold


		results.append(properties)
		
	return results

def evaluate_noise_chunks(pairs, signals, peaks, distances, method="median", height_multiplier=0.75):
	# make sure index is correct
	signals = signals.reset_index()
	signals.index = signals["TF1"] + "-" + signals["TF2"]

	peaks = peaks.reset_index()
	peaks.index = peaks["TF1"] + "-" + peaks["TF2"]

	# get data column
	distance_cols = np.array([-1 if d == "neg" else d for d in distances]) #neg counts as -1
	results = []
	for pair in pairs:
		# get pair
		tf1, tf2 = pair
		ind = "-".join(pair)
		signal = signals.loc[ind].loc[distance_cols].values
		
		# get peaks for specific pair
		peaks_pair = peaks[(peaks.TF1 == tf1) & (peaks.TF2 == tf2)]

		results.append([tf1, tf2, _get_noise_measure(peaks_pair, signal, method, height_multiplier)])
		
	return results



def _get_noise_measure(peaks, signal, method, height_multiplier):
	#check method input
	check_string(method, ["median", "min_max"], "method")

	# get the cutting points fot the signal
	cuts = _get_cut_points(peaks, height_multiplier, signal)

	# cut all peaks out of the signal
	for cut in cuts:
		signal[cut[0]:cut[1]] = np.nan

	measure = None
	if method == "median":
		measure = pd.Series(signal).median()
	elif method == "min_max":
		measure = signal.max() - signal.min()
	
	return float(measure)

def _get_cut_points(peaks, height_multiplier, signal):
	cuts =[]
	for idx,row in peaks.iterrows():
		# get the peak distance
		peak = row.Distance
		# get the peak height 
		peak_height = signal[peak]
		# determine cutoff, in common sense this should be "going ~25% down the peak size"
		cut_off = height_multiplier * peak_height
		cuts.append(_expand_peak(peak, cut_off, signal))
	return cuts

def _expand_peak(start_pos, cut_off, signal):
	found_left = False
	found_right = False
	pos_left = start_pos - 1
	pos_right = start_pos + 1

	# expand the peak until both borders are found
	while(not(found_left & found_right)):
		# left side
		if(not found_left):
			# left border not found
			if pos_left <= -1: # check if position less than start of signal
				found_left = True
				left = 0
			elif signal[pos_left] <= cut_off:
				found_left = True
				left = pos_left  + 1 # we are one to far left
			pos_left -= 1
		
		# right side
		if(not found_right):
			# right border not found	
			if  pos_right == len(signal): # check if position higher than end of signal
				found_right = True
				right = len(signal) - 1
			elif signal[pos_right] < cut_off:
				found_right = True
				right = pos_right - 1 # we are one to far right
			pos_right += 1
	return(left, right)

def fast_rolling_mean(arr, w):
	"""
	Adaption of tobias.signals.fast_rolling_math to avoid NaN in flanking positions
	Rolling operation of arr with window size w 
	"""

	lf = int(np.ceil( (w-1) / 2.0))
	rf = int(np.floor( (w-1) / 2.0))
	#Expand the array with the first value to the left 
	arr = np.concatenate((np.repeat(arr[0], lf), arr))
	#Expand the array with the last value to the right 
	arr = np.concatenate((arr, np.repeat(arr[0], rf)))

	# use fast_rolling_math from tobias.utils.signals
	roll_arr = fast_rolling_math(arr.astype(float), w, "mean")
	

	#remove nan's ( artifical new flanks)
	roll_arr = roll_arr[~np.isnan(roll_arr)]

	return roll_arr
