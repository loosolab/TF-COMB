from kneed import DataGenerator, KneeLocator
from tobias.utils.regions import OneRegion, RegionList
from tobias.utils.motifs import MotifList
from copy import deepcopy
import pandas as pd
import statsmodels.stats.multitest
import numpy as np
import scipy.stats
import pysam
import random
import string
import multiprocessing as mp
import os

#----------------- Minimal TFBS class based on the TOBIAS 'OneRegion' class -----------------#

class OneTFBS():

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

#--------------------------------- File/type checks ---------------------------------#

def check_columns(df, columns):
	""" Utility to check whether columns are found within a pandas dataframe """
	
	df_columns = df.columns

	for column in columns:
		if column is not None:
			if column not in df_columns:
				raise ValueError("Column '{0}' is not found in dataframe. Available columns are: {1}".format(column, df_columns))

def check_writeability(file_path):
	""" Check if a file is writeable """

	#Check if file already exists
	if os.path.exists(file_path):
		if not os.path.isfile(file_path): # is it a file or a dir?
			raise ValueError("Path '{0}' is not a file".format(file_path))

	#check writeability of parent dir
	else:
		pdir = os.path.dirname(file_path)
		if os.access(pdir, os.W_OK) == False:
			raise ValueError("File '{0}' within dir {1} is not writeable".format(file_path, pdir))


def check_type(obj, types, name=None):
	"""
	Check whether given object is within a list of allowed types.

	Parameters
	----------
	obj : object
		Object to check type on
	types : list
		A list of allowed object types
	name : str, optional
		Name of object to be written in error. Default: None (the input is referred to as 'object')

	Raises
	--------
	ValueError
		If object type is not within types.
	"""
	
	#Check if any of the types fit
	flag = 0
	for t in types:
		if isinstance(obj, t):
			flag = 1

	#Raise valueError if none of the types fit
	if flag == 0:	
		name = "object" if name is None else '\'{0}\''.format(name)
		raise ValueError("The {0} given has type '{1}', but must be one of: {2}".format(name, type(obj), types))

def check_string(astring, allowed):
	""" 
	Check whether given string is within a list of allowed strings.
	
	Parameters
	----------
	astring : str
		A string to check.
	allowed : str or list of strings
		A string or list of allowed strings to check against 'astring'.
	"""

def check_value(value, vmin=-np.inf, vmax=np.inf, integer=False):
	"""
	value : int or float
		The value to check against vmin/max.
	vmin : int or float, optional
		Default: -infinity
	vmax : int or float
		Default: +infinity
	integer : bool, optional
		Value must be an integer. Default: False.
	"""

	if integer == True:
		#check if value is integer
		pass

	else:
		#check if value is any value
		pass

	if (value >= vmin) & (value <= vmax):
		pass


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
	""" Read motifs from motifs_file """

	#Read and prepare motifs
	motifs_obj = MotifList().from_file(motifs_file)

	_ = [motif.get_threshold(motif_pvalue) for motif in motifs_obj]
	_ = [motif.set_prefix(motif_naming) for motif in motifs_obj] #using naming from args

	return(motifs_obj)

def open_genome(genome_f):	
	""" """
	genome_obj = pysam.FastaFile(genome_f)
	return(genome_obj)

def calculate_TFBS(regions, motifs, genome):
	"""
	Multiprocessing-safe function to scan for motif occurrences

	Parameters
	----------
	genome : str or 
		If string , genome will be opened 
	
	regions : RegionList()
		A RegionList() object of regions 

	Returns
	----------
	List of TFBS within regions

	"""

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

	if isinstance(genome, str):
		genome_obj.close()

	return(TFBS_list)


def remove_duplicates(TFBS):
	""" """

	filtered = TFBS

	return(filtered)


def resolve_overlapping(TFBS):
	""" Remove self-overlapping regions """

	#Split TFBS into dict per name
	sites_per_name = {}
	for site in TFBS:
		if site.name not in sites_per_name:
			sites_per_name[site.name] = RegionList()
		sites_per_name[site.name].append(site)

	#Resolve overlaps
	resolved = RegionList()
	for name in sites_per_name:
		resolved.extend(sites_per_name[name].resolve_overlaps())
	resolved.loc_sort()
	
	return(resolved)

#--------------------------------- P-value calculation ---------------------------------#

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

	Returns
	--------
	List of p-values in order of input table

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
	p = Progress(n_jobs, 10, logger)

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

	#Adjust pvalues
	table[col + "_adj"] = statsmodels.stats.multitest.multipletests(table[col], method="Bonferroni")[1]

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
		Value to remove. 
		
	n : int
		The number of elements in the distribution (including value)
	
	returns()
	"""

	#Calculate new mean
	bg_mean = (mu*n - value)/(n-1)

	#Calculate new std
	var = std**2
	bg_var = n/(n-1) * (var - (mu - value)**2/(n-1))
	bg_std = np.sqrt(bg_var)

	bg_n = n - 1

	return((bg_mean, bg_std, bg_n))


#--------------------------------- Working with TF-comb objects ---------------------------------#

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
