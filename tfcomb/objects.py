#objects.py: Contains CombObj, DiffCombObj and DistObj classes
#
#@author: Mette Bentsen and Vanessa Heger
#@contact: mette.bentsen (at) mpi-bn.mpg.de
#@license: MIT


import os 
import pandas as pd
import itertools
import multiprocessing as mp
import numpy as np
import copy
import glob
import fnmatch
import dill
import collections
import math
import re

#Statistics
import qnorm #quantile normalization
from scipy.stats import linregress
from kneed import KneeLocator
import statsmodels.api as sm
from sklearn.mixture import GaussianMixture
from scipy import stats

#Modules for plotting
import matplotlib.pyplot as plt
import seaborn as sns
from tfcomb import utils

#Utilities from TOBIAS
import tobias
from tobias.utils.motifs import MotifList
from tobias.utils.regions import OneRegion, RegionList
#from tobias.utils.signals import fast_rolling_math

#TF-comb modules
import tfcomb
import tfcomb.plotting
import tfcomb.network
import tfcomb.analysis
from tfcomb.counting import count_co_occurrence
from tfcomb.logging import TFcombLogger, InputError
from tfcomb.utils import OneTFBS, Progress, check_type, check_value, check_string, unique_region_names, check_columns, check_writeability
from tfcomb.utils import *


np.seterr(all='raise') # raise errors for runtimewarnings
#pd.options.mode.chained_assignment = 'raise' # for debugging
pd.options.mode.chained_assignment = None

class CombObj(): 
	"""
	The main class for collecting and working with co-occurring TFs.

	Examples
	----------

	>>> C = tfcomb.objects.CombObj()

	# Verbosity of the output log can be set using the 'verbosity' parameter: \n

	>>> C = tfcomb.objects.CombObj(verbosity=2)

	"""

	#-------------------------------------------------------------------------------#
	#------------------------------- Getting started -------------------------------#
	#-------------------------------------------------------------------------------#

	def __init__(self, verbosity=1): #set verbosity 

		#Function and run parameters
		self.verbosity = verbosity  #0: error, 1:info, 2:debug, 3:spam-debug
		self.logger = TFcombLogger(self.verbosity)
		
		#Variables for storing input data
		self.prefix = None 	     #is used when objects are added to a DiffCombObj
		self.TFBS = None		 #None or filled with list of TFBS
		self.TF_names = []		 #list of TF names

		#Variables for counts
		self.TF_counts = None 	 #numpy array of size n_TFs
		self.pair_counts = None	 #numpy matrix of size n_TFs x n_TFs
		self.bg_counts_mean = None #numpy matrix of size n_TFs x n_TFs
		self.bg_counts_std = None  #numpy matrix of size n_TFs x n_TFs

		#Market basket and network analysis
		self.rules = None  		 #filled in by .market_basket()
		self.network = None		 #filled by .build_network()

		#Default parameters
		self.min_dist = 0
		self.max_dist = 100
		self.directional = False
		self.min_overlap = 0
		self.max_overlap = 0
		self.anchor = "inner"
		self.anchor_to_int = {"inner": 0, "outer": 1, "center": 2}

		#Variable for storing DistObj for distance analysis
		self.distObj = None

	def __str__(self):
		""" Returns a string representation of the CombObj depending on what variables are already stored """
		
		s = "<CombObj"

		if self.TFBS is not None:
			s += ": {0} TFBS ({1} unique names)".format(len(self.TFBS), len(self.TF_names)) 

			if self.rules is not None:
				s += " | Market basket analysis: {0} rules".format(self.rules.shape[0])

		s += ">"
		return(s)

	def __repr__(self):
		return(self.__str__())
	
	def __getstate__(self):
		""" Get state for pickling"""

		return {k:v for (k, v) in self.__dict__.items() if k != "_sites"} #do not pickle the internal _sites array 

	def __add__(self, obj):
		"""
		Internal method to add two CombObj together using: `CombObj1 + CombObj2 = new_CombObj`. This merges the .TFBS of both objects under the hood.
		
		Parameters:
		----------
		obj : CombObj 
			A CombObj to add.

		Returns:
		----------
			A merged CombObj.
		"""

		combined = CombObj(self.verbosity) #initialize empty combobj

		#Merge TFBS
		combined.TFBS = RegionList(self.TFBS + obj.TFBS)
		combined.TFBS.loc_sort() 				#sort TFBS by coordinates

		#Set .TF_names of the new list
		counts = {r.name: "" for r in combined.TFBS}
		combined.TF_names = sorted(list(set(counts.keys()))) #ensures that the same TF order is used across cores/subsets		

		return(combined)
	
	def copy(self):
		""" Returns a deep copy of the CombObj """

		copied = copy.deepcopy(self)
		copied.logger = TFcombLogger(self.verbosity) #receives its own logger
		return(copied)
	
	def set_verbosity(self, level):
		""" Set the verbosity level for logging after creating the CombObj.

		Parameters
		----------
		level : int
			A value between 0-3 where 0 (only errors), 1 (info), 2 (debug), 3 (spam debug). Default: 1.
		"""

		self.verbosity = level
		self.logger = TFcombLogger(self.verbosity) #restart logger with new verbosity

	def set_prefix(self, prefix):
		""" Sets the .prefix variable of the object. Useful when comparing two objects in a DiffCombObj. 
		
		Parameters
		-----------
		prefix : str
			A string to add as .prefix for this object, e.g. 'control', 'treatment' or 'analysis1'. 
		"""

		check_type(prefix, str, "prefix")
		self.prefix = prefix

	#-------------------------------------------------------------------------------#
	#----------------------------- Checks for the object----------------------------#
	#-------------------------------------------------------------------------------#

	def _check_TFBS(self):
		""" Internal check whether the .TFBS was already filled. Raises InputError when .TFBS is not available."""

		#Check that TFBS exist and that it is RegionList
		if self.TFBS is None or (not isinstance(self.TFBS, list) and not isinstance(self.TFBS, RegionList)):
			raise InputError("No TFBS available in '.TFBS'. The TFBS are set either using .TFBS_from_motifs, .TFBS_from_bed or TFBS_from_TOBIAS.")

	def _check_counts(self):
		""" Internal check whether .count_within was already run. Raises InputError when counts are not available or if counts have the wrong dimensions."""

		#Check if counts were set
		attributes = ["TF_counts", "pair_counts"]
		for att in attributes:
			val = getattr(self, att)
			if val is None:
				raise InputError(f"Internal counts for '{att}' were not set. Please run .count_within() to obtain TF-TF co-occurrence counts.")
			else:
				n_TFs = len(self.count_names)	#can be different than self.TF_names if stranded == True
				tfcomb.utils.check_type(val, np.ndarray, val) #raises inputerror if val is not None, but also not array
				size = val.shape
				
				invalid = 0
				if len(size) == 1:
					if size != (n_TFs,):
						invalid = 1
				elif len(size) == 2:
					if size != (n_TFs,n_TFs):
						invalid = 1

				#If the sizes of arrays do not fit number of TFs
				if invalid == 1:
					err = f"Internal counts for '{att}' had shape {size} which does not fit with length of TF names ({n_TFs})."
					err += "Please check that .count_within() was run on the same TFs as currently set within the object"
					raise InputError(err)

	def _check_rules(self):
		""" Internal check whether .rules were filled. Raises InputError when .rules are not available. """

		if self.rules is None or not isinstance(self.rules, pd.DataFrame):
			raise InputError("No market basket rules found in .rules. The rules are found by running .market_basket().")
	
	def check_pair(self, pair):
		""" Checks if a pair is valid and present. 
		
		Parameters
		----------
		pair : tuple(str,str)
			TF names for which the test should be performed. e.g. ("NFYA","NFYB")
		
		Raises
		----------
		"""

		#check member size
		if len(pair) != 2:
			raise InputError(f'{pair} is not valid. It should contain exactly two TF names per pair. e.g. ("NFYA","NFYB")')
		
		# check tf names are string
		tf1,tf2 = pair 
		tfcomb.utils.check_type(tf1, str, "TF1 from pair")
		tfcomb.utils.check_type(tf2, str, "TF2 from pair")

		# check rules are filled
		if type(self) == DistObj and self.rules is None:
			raise InputError(".rules not filled. Please run .fill_rules() first.")

		# check tf1 is present within object
		if tf1 not in self.TF_names:
			raise InputError(f"{tf1} (TF1) is not valid as it is not present in the current object.")

		# check tf1 is present within object
		if tf2 not in self.TF_names:
			raise InputError(f"{tf2} (TF2) is not valid as it is not present in the current object.")
		
		if type(self) == DistObj:
			if len(self.rules.loc[((self.rules["TF1"] == tf1) & (self.rules["TF2"] == tf2))]) == 0:
				raise InputError(f"No rules for pair {tf1} - {tf2} found.")

	#-------------------------------------------------------------------------------#
	#----------------------------- Save / import object ----------------------------#
	#-------------------------------------------------------------------------------#

	def to_pickle(self, path):
		""" Save the CombObj to a pickle file.
		
		Parameters
		----------
		path : str
			Path to the output pickle file e.g. 'my_combobj.pkl'.

		See also
		---------
		from_pickle
		"""

		f_out = open(path, 'wb') 
		dill.dump(self, f_out)


	def from_pickle(self, path):
		"""
		Import a CombObj from a pickle file.

		Parameters
		-----------
		path : str
			Path to an existing pickle file to read.

		Raises
		-------
		InputError
			If read object is not an instance of CombObj.
		
		See also
		----------
		to_pickle
		"""

		filehandler = open(path, 'rb') 

		try:
			obj = dill.load(filehandler)
		except AttributeError as e:
			if "new_block" in str(e):
				s = f"It looks like the CombObj was built with pandas 1.3.x, but the current pandas version is {pd.__version__}."
				s += " Please rebuild the CombObj with pandas 1.2.x or upgrade pandas to 1.3.x to load the object."
				raise InputError(s)
			else: #another error during reading
				raise e	

		#Check if object is CombObj
		if not isinstance(obj, CombObj):
			raise InputError("Object from '{0}' is not a CombObj".format(path))

		#Overwrite self with CombObj
		self = obj
		self.set_verbosity(self.verbosity) #restart logger
		
		return(self)

	#-------------------------------------------------------------------------------#
	#-------------------------- Setting up the .TFBS list --------------------------#
	#-------------------------------------------------------------------------------#

	def TFBS_from_motifs(self, regions, 
								motifs, 
								genome,
								motif_pvalue=1e-05,
								motif_naming="name",
								gc=0.5, 
								resolve_overlapping="merge", 
								extend_bp=0,
								threads=1, 
								overwrite=False,
								_suffix=""): #suffix to add to output motif names

		"""
		Function to calculate TFBS from motifs and genome fasta within the given genomic regions.

		Parameters
		-----------
		regions : str or tobias.utils.regions.RegionList
			Path to a .bed-file containing regions or a tobias-format RegionList object. 
		motifs : str or tobias.utils.motifs.MotifList
			Path to a file containing JASPAR/MEME-style motifs or a tobias-format MotifList object.
		genome : str
			Path to the genome fasta-file to use for scan.
		motif_pvalue : float, optional
			The pvalue threshold for the motif search. Default: 1e-05.
		motif_naming : str, optional
			How to name TFs based on input motifs. Must be one of: 'name', 'id', 'name_id' or 'id_name'. Default: "name".
		gc : float between 0-1, optional
			Set GC-content for the motif background model. Default: 0.5.
		resolve_overlapping : str, optional
			Control how to treat overlapping occurrences of the same TF. Must be one of "merge", "highest_score" or "off". If "highest_score", the highest scoring overlapping site is kept.
			If "merge", the sites are merged, keeping the information of the first site. If "off", overlapping TFBS are kept. Default: "merge".
		extend_bp : int, optional
			Extend input regions with 'extend_bp' before scanning. Default: 0.
		threads : int, optional
			How many threads to use for multiprocessing. Default: 1. 
		overwrite : boolean, optional
			Whether to overwrite existing sites within .TFBS. Default: False (sites are appended to .TFBS).

		Returns
		-----------
		None 
			.TFBS_from_motifs fills the objects' .TFBS variable

		"""
		
		#Check input validity
		allowed_motif_naming = ["name", "id", "name_id", "id_name"]
		check_type(regions, [str, tobias.utils.regions.RegionList], "regions")
		check_type(motifs, [str, tobias.utils.motifs.MotifList], "motifs")
		check_type(genome, str, "genome")
		check_value(motif_pvalue, vmin=0, vmax=1, name="motif_pvalue")
		check_string(motif_naming, allowed_motif_naming, "motif_naming")
		check_value(gc, vmin=0, vmax=1, name="gc")
		check_string(resolve_overlapping, ["off", "merge", "highest_score"], "resolve_overlapping")
		check_value(threads, vmin=1, name="threads")
		check_type(overwrite, bool, "overwrite")
		
		#If previous TFBS should be overwritten or TFBS should be initialized
		initialized = 0
		if overwrite == True or self.TFBS is None:
			initialized = 1
			self.TFBS = RegionList()
			self.TF_names = []

		motifs = copy.deepcopy(motifs)  #ensures that original motifs are not altered

		#Setup regions
		if isinstance(regions, str):
			regions_f = regions
			regions = RegionList().from_bed(regions)
			self.logger.debug("Read {0} regions from {1}".format(len(regions), regions_f))

		#Extend input regions
		if extend_bp > 0:
			for region in regions:
				region.extend_reg(extend_bp)

		#Setup motifs
		if isinstance(motifs, str):
			motifs_f = motifs
			motifs = tfcomb.utils.prepare_motifs(motifs_f, motif_pvalue, motif_naming)
			self.logger.debug("Read {0} motifs from '{1}'".format(len(motifs), motifs_f))
		else:
			
			#set gc for motifs
			#bg = 

			_ = [motif.get_threshold(motif_pvalue) for motif in motifs]
			_ = [motif.set_prefix(motif_naming) for motif in motifs]


		#Set suffix to motif names
		for motif in motifs:
			motif.prefix = motif.prefix + _suffix

		#Check that regions are within the genome bounds
		genome_obj = tfcomb.utils.open_genome(genome)
		check_boundaries(regions, genome_obj)
		genome_obj.close()

		#Scan for TFBS (either single-thread or with multiprocessing)
		self.logger.info("Scanning for TFBS with {0} thread(s)...".format(threads))
		if threads == 1:

			n_regions = len(regions)
			chunks = regions.chunks(100) 
			genome_obj = tfcomb.utils.open_genome(genome)	#open pysam fasta obj

			n_regions_processed = 0
			TFBS = RegionList([])	#initialize empty list	
			for region_chunk in chunks:

				region_TFBS = tfcomb.utils.calculate_TFBS(region_chunk, motifs, genome_obj, resolve_overlapping)
				TFBS += region_TFBS

				#Update progress
				n_regions_processed += len(region_chunk)
				self.logger.debug("{0:.1f}% ({1} / {2})".format(n_regions_processed/n_regions*100, n_regions_processed, n_regions))

			genome_obj.close()

		else:
			chunks = regions.chunks(100) #creates chunks of regions for multiprocessing

			#Setup pool
			pool = mp.Pool(threads)
			jobs = []
			for region_chunk in chunks:
				self.logger.spam("Starting job for region_chunk of length: {0}".format(len(region_chunk)))
				jobs.append(pool.apply_async(tfcomb.utils.calculate_TFBS, (region_chunk, motifs, genome, resolve_overlapping)))
			pool.close()
			
			log_progress(jobs, self.logger) #only exits when the jobs are complete
			
			results = [job.get() for job in jobs]
			pool.join()

			#Join all TFBS to one list
			TFBS = RegionList(sum(results, []))

		self.logger.info("Processing scanned TFBS")

		#Join and process TFBS
		self.TFBS += TFBS
		self.TFBS.loc_sort()
		TF_names = unique_region_names(TFBS)
		self.TF_names = sorted(list(set(self.TF_names + TF_names)))

		#Resolve any leftover overlaps between jobs
		if resolve_overlapping != "off":
			TFBS = tfcomb.utils.resolve_overlaps(TFBS, how=resolve_overlapping)

		#Final info log
		self.logger.info("Identified {0} TFBS ({1} unique names) within given regions".format(len(TFBS), len(TF_names)))
		if initialized == 0: #if sites were added, log the updated
			self.logger.info("The attribute .TFBS now contains {0} TFBS ({1} unique names)".format(len(self.TFBS), len(self.TF_names)))

	def TFBS_from_bed(self, bed_file, overwrite=False):
		"""
		Fills the .TFBS attribute using a precalculated set of binding sites e.g. from ChIP-seq.

		Parameters
		-------------
		bed_file : str 
			A path to a .bed-file with precalculated binding sites. The 4th column of the file should contain the name of the TF in question.
		overwrite : boolean
			Whether to overwrite existing sites within .TFBS. Default: False (sites are appended to .TFBS).

		Returns
		-----------
		None 
			The .TFBS variable is filled in place
		"""

		#Check input parameters
		check_type(bed_file, str, "bed_file")
		check_type(overwrite, bool, "overwrite")

		#If previous TFBS should be overwritten or TFBS should be initialized
		initialized = 0
		if overwrite == True or self.TFBS is None:
			self.TFBS = RegionList()
			self.TF_names = []
			initialized = 1

		#Read sites from file
		self.logger.info("Reading sites from '{0}'...".format(bed_file))
		read_TFBS = RegionList([OneTFBS(region) for region in RegionList().from_bed(bed_file)])
		
		#Add TFBS to internal .TFBS list and process
		self.logger.info("Processing sites")
		self.TFBS += read_TFBS
		self.TFBS.loc_sort()
		read_TF_names = unique_region_names(read_TFBS)
		self.TF_names = sorted(list(set(self.TF_names + read_TF_names)))

		#Final info log
		self.logger.info("Read {0} sites ({1} unique names)".format(len(read_TFBS), len(read_TF_names)))
		if initialized == 0: #if sites were added, log the updated
			self.logger.info("The attribute .TFBS now contains {0} TFBS ({1} unique names)".format(len(self.TFBS), len(self.TF_names)))

	def TFBS_from_TOBIAS(self, bindetect_path, condition, overwrite=False):
		"""
		Fills the .TFBS variable with pre-calculated bound binding sites from TOBIAS BINDetect.

		Parameters
		-----------
		bindetect_path : str
			Path to the BINDetect-output folder containing <TF1>, <TF2>, <TF3> (...) folders.
		condition : str
			Name of condition to use for fetching bound sites.
		overwrite : boolean
			Whether to overwrite existing sites within .TFBS. Default: False (sites are appended to .TFBS).
		
		Returns
		-----------
		None 
			The .TFBS variable is filled in place

		Raises
		-------
		InputError 
			If no files are found in path or if condition is not one of the avaiable conditions.
		"""

		#Check input
		check_type(bindetect_path, str, "bindetect_path")
		check_type(condition, str, "condition")
		check_type(overwrite, bool, "overwrite")

		#If previous TFBS should be overwritten or TFBS should be initialized
		initialized = 0
		if overwrite == True or self.TFBS is None:
			self.TFBS = RegionList()
			self.TF_names = []
			initialized = 1

		#Grep for files within given path
		pattern = os.path.join(bindetect_path, "*", "beds", "*_bound.bed")
		files = glob.glob(pattern)
		if len(files) == 0:
			raise InputError("No '_bound'-files were found in path. Please ensure that the given path is the output of TOBIAS BINDetect.")
		
		#Check if condition given is within available_conditions
		available_conditions = set([os.path.basename(f).split("_")[-2] for f in files])
		if condition not in available_conditions:
			raise InputError("Condition must be one of: {0}".format(list(available_conditions)))

		#Read sites from files
		condition_pattern = os.path.join(bindetect_path, "*", "beds", "*" + condition + "_bound.bed")
		condition_files = fnmatch.filter(files, condition_pattern)

		TFBS = RegionList()
		for f in condition_files:
			TFBS += RegionList([OneTFBS(region) for region in RegionList().from_bed(f)])

		#Add TFBS to internal .TFBS list and process
		self.TFBS += TFBS
		self.TFBS.loc_sort()
		read_TF_names = unique_region_names(TFBS)
		self.TF_names = sorted(list(set(self.TF_names + read_TF_names)))

		self.logger.info("Read {0} sites ({1} unique names) from condition '{2}'".format(len(TFBS), len(read_TF_names), condition))
		if initialized == 0:
			self.logger.info("The attribute .TFBS now contains {0} TFBS ({1} unique names)".format(len(self.TFBS), len(self.TF_names)))
		
	#-------------------------------------------------------------------------------------------------------------#
	#----------------------------------- Filtering and processing of TFBS ----------------------------------------#
	#-------------------------------------------------------------------------------------------------------------#

	def cluster_TFBS(self, threshold=0.5, merge_overlapping=True):
		""" 
		Cluster TFBS based on overlap of individual binding sites. This can be used to pre-process motif-derived TFBS into TF "families" of TFs with similar motifs.
		This changes the .name attribute of each site within .TFBS to represent the cluster (or the original TF name if no cluster was found).
		
		Parameters
		------------
		threshold : float from 0-1, optional
			The threshold to set when clustering binding sites. Default: 0.5.
		merge_overlapping : bool, optional
			Whether to merge overlapping sites following clustering. 
			If True, overlapping sites from the same cluster will be merged to one site (spanning site1-start -> site2-end). 
			If False, the original sites (but with updated names) will be kept in .TFBS. Default: True.

		Returns
		--------
		None 
			The .TFBS names are updated in place.
		"""

		#Check given input
		self._check_TFBS()
		tfcomb.utils.check_value(threshold, vmin=0, vmax=1, name="threshold")
		tfcomb.utils.check_type(merge_overlapping, bool, "merge_overlapping")

		#Calculate overlap of TFBS
		overlap_dict = self.TFBS.count_overlaps()
		
		#Setup cluster object
		clust = tobias.utils.regions.RegionCluster(overlap_dict)
		clust.cluster(threshold=threshold)

		#Create name -> new name dict
		conversion = {}
		for cluster_idx in clust.clusters:
			
			members = clust.clusters[cluster_idx]["member_names"]

			if len(members) > 1:
				new_name = "/".join(sorted(members))
				for member in members:
					conversion[member] = new_name
			else:
				conversion[members[0]] = members[0]

		#Convert TFBS names
		for site in self.TFBS:
			site.name = conversion[site.name]
		
		#Handle overlapping sites within the same clusters
		if merge_overlapping: 
			n_sites_orig = len(self.TFBS)
			self.TFBS = tfcomb.utils.merge_self_overlaps(self.TFBS)
			self.logger.info("merge_overlapping == True: Merged {0} .TFBS to {1} non-overlapping sites".format(n_sites_orig, len(self.TFBS)))

		#Update names of TFBS
		n_names_orig = len(self.TF_names) 				#number of unique names before clustering
		self.TF_names = unique_region_names(self.TFBS)	#unique names after clustering
		n_names_clustered = len(self.TF_names)

		self.logger.info("TFBS were clustered from {0} to {1} unique names. The new TF names can be seen in <CombObj>.TFBS and <CombObj>.TF_names.",format(n_names_orig, n_names_clustered))


	def subset_TFBS(self, names=None, 
						  regions=None):
		"""
		Subset .TFBS in object to specific regions or TF names. Can be used to select only a subset of TFBS (e.g. only in promoters) to run analysis on. Note: Either 'names' or 'regions' must be given - not both.

		Parameters
		-----------
		names : list of strings, optional
			A list of names to keep. Default: None.
		regions : str or RegionList, optional
			Path to a .bed-file containing regions or a tobias-format RegionList object. Default: None.

		Returns
		-------
		None
			The .TFBS attribute is updated in place.
		"""

		#Check given input
		self._check_TFBS()
		if (names is None and regions is None) or (names is not None and regions is not None):
			raise InputError("You must give either 'names' or 'regions' to .subset_TFBS.")

		#Subset TFBS based on input
		if regions is not None:
			tfcomb.utils.check_type(regions, [str, tobias.utils.regions.RegionList], "regions")

			#If regions are string, read to internal format
			if isinstance(regions, str):
				regions = RegionList().from_bed(regions)
			
			#Get indices of overlapping sites
			n_TFBS = len(self.TFBS)
			self.logger.info("Overlapping {0} TFBS with {1} regions".format(n_TFBS, len(regions)))
			
			TFBS_overlap_labeled = tfcomb.utils.add_region_overlap(self.TFBS, regions, att="overlap") #adds overlap boolean to TFBS
			idx = [i for i, site in enumerate(TFBS_overlap_labeled) if site.overlap == True]
			self.TFBS = RegionList([self.TFBS[i] for i in idx])
		
		elif names is not None:
			tfcomb.utils.check_type(names, [list, set, tuple], "names")

			#Check that strings overlap with TFBS
			names = set(names)			  #input names to overlap
			TF_names = set(self.TF_names) #names from object
			not_in_TFBS = names - TF_names
			if len(not_in_TFBS) > 0:
				self.logger.warning("{0} names from 'names' were not found in <CombObj> names and could therefore not be selected. These names are: {1}".format(len(not_in_TFBS), not_in_TFBS))
			in_TFBS = TF_names.intersection(names)

			if len(in_TFBS) == 0:
				raise InputError("No overlap found between 'names' and names from <CombObj>.TFBS. Please select names of TFs within the data.")

			self.TFBS = RegionList([site for site in self.TFBS if site.name in in_TFBS])
			self.TF_names = unique_region_names(self.TFBS)	#unique names after clustering

		self.logger.info("Subset finished! The attribute .TFBS now contains {0} sites.".format(len(self.TFBS)))

	def TFBS_to_bed(self, path):
		"""
		Writes out the .TFBS regions to a .bed-file. This is a wrapper for the tobias.utils.regions.RegionList().write_bed() utility.

		Parameters
		----------
		path : str
			File path to write .bed-file to.
		"""

		#Check input
		self._check_TFBS()
		check_writeability(path)

		#Call the .write_bed utility from tobias
		f = open(path, "w")
		s = "\n".join([str(site) for site in self.TFBS]) + "\n"
		f.write(s)
		f.close()

	def _prepare_TFBS(self, force=False):
		""" 
			Prepare the TFBS for internal counting within count_within. Sets the internal ._sites attribute. 
			Checks the existence and correct length of _sites, and only creates it if not already saved. Set 'force' to True to force recalculation.
		"""

		#Find out if _sites should be set
		create = 1
		if hasattr(self, "_sites"):
			if len(self.TFBS) == len(self._sites):
				create = 0
			else:
				self.logger.debug("Length of .TFBS ({0}) is different than existing ._sites ({1}) attribute. Recreating ._sites.".format(len(self.TFBS), len(self._sites)))	

		if create == 1 or force == True or not hasattr(self, "name_to_idx"):

			self.logger.info("Setting up binding sites for counting")
			chromosomes = {site.chrom:"" for site in self.TFBS}.keys()
			chrom_to_idx = {chrom: idx for idx, chrom in enumerate(chromosomes)}
			self.name_to_idx = {name: idx for idx, name in enumerate(self.TF_names)}
			self._sites = np.array([(chrom_to_idx[site.chrom], site.start, site.end, self.name_to_idx[site.name]) for site in self.TFBS]) #numpy integer array

	@staticmethod
	def _get_sort_idx(sites, anchor="center"):
		""" Get indices for sorting sites (from _prepare_TFBS) depending on anchor.
		
		Parameters
		-----------
		sites : np.array
			The ._sites array prepared by _prepare_TFBS.
		anchor : str
			The anchor for calculating distances. One of "inner", "outer" or "center". Default: "center".
		"""
		
		if anchor == "center":
			sort_idx = [i for i, _ in sorted(enumerate(sites), key=lambda tup: (tup[1][0], int((tup[1][1] + tup[1][2]) / 2)))]
		
		elif anchor == "inner" or anchor == "outer":
			sort_idx = [i for i, _ in sorted(enumerate(sites), key=lambda tup: (tup[1][0], tup[1][1], tup[1][2]))]

		return sort_idx

	#-------------------------------------------------------------------------------------------------------------#
	#----------------------------------------- Counting co-occurrences -------------------------------------------#
	#-------------------------------------------------------------------------------------------------------------#

	def count_within(self, min_dist=0, 
						   max_dist=100, 
						   min_overlap=0,
						   max_overlap=0, 
						   stranded=False, 
						   directional=False, 
						   binarize=False,
						   anchor="inner",
						   n_background=50,
						   threads=1):
		""" 
		Count co-occurrences between TFBS. This function requires .TFBS to be filled by either `TFBS_from_motifs`, `TFBS_from_bed` or `TFBS_from_tobias`. 
		This function can be followed by .market_basket to calculate association rules.
		
		Parameters
		-----------
		min_dist : int
			Minimum distance between two TFBS to be counted as co-occurring. Distances are calculated depending on the 'anchor' given. Default: 0.
		max_dist : int
			Maximum distance between two TFBS to be counted as co-occurring. Distances are calculated depending on the 'anchor' given. Default: 100.
		min_overlap : float between 0-1, optional
			Minimum overlap fraction needed between sites, e.g. 0 = no overlap needed, 1 = full overlap needed. Default: 0.
		max_overlap : float between 0-1, optional
			Controls how much overlap is allowed for individual sites. A value of 0 indicates that overlapping TFBS will not be saved as co-occurring. 
			Float values between 0-1 indicate the fraction of overlap allowed (the overlap is always calculated as a fraction of the smallest TFBS). A value of 1 allows all overlaps. Default: 0 (no overlap allowed).
		stranded : bool
			Whether to take strand of TFBSs into account. Default: False.
		directional : bool
			Decide if direction of found pairs should be taken into account, e.g. whether  "<---TF1---> <---TF2--->" is only counted as 
			TF1-TF2 (directional=True) or also as TF2-TF1 (directional=False). Default: False.
		binarize : bool, optional
			Whether to count a TF1-TF2 more than once per window (e.g. in the case of "<TF1> <TF2> <TF2> (...)"). Default: False.
		anchor : str, optional
			The anchor to use for calculating distance. Must be one of ["inner", "outer", "center"]
		n_background : int, optional
			Number of random co-occurrence backgrounds to obtain. This number effects the runtime of .count_within, but 'threads' can be used to speed up background calculation. Default: 50.
		threads : int, optional
			Number of threads to use. Default: 1.

		Returns
		----------
		None 
			Fills the object variables .TF_counts and .pair_counts.
		
		Raises
		--------
		ValueError
			If .TFBS has not been filled.
		"""

		anchor_str_to_int = {"inner": 0, "outer": 1, "center": 2}

		#Check input parameters
		self._check_TFBS()
		tfcomb.utils.check_value(min_dist, vmin=0, integer=True, name="min_dist")
		tfcomb.utils.check_value(max_dist, vmin=min_dist, integer=True, name="max_dist")
		tfcomb.utils.check_value(min_overlap, vmin=0, vmax=1, name="min_overlap")
		tfcomb.utils.check_value(max_overlap, vmin=0, vmax=1, name="max_overlap")
		tfcomb.utils.check_type(stranded, bool, "stranded")
		tfcomb.utils.check_type(directional, bool, "directional")
		tfcomb.utils.check_type(binarize, bool, "binarize")
		tfcomb.utils.check_string(anchor, list(anchor_str_to_int.keys()), "anchor")

		#Update object variables
		self.rules = None 	#Remove .rules if market_basket() was previously run
		self.n_TFBS = len(self.TFBS)	#number of TFBS counted
		self.min_dist = min_dist
		self.max_dist = max_dist
		self.min_overlap = min_overlap
		self.max_overlap = max_overlap
		self.stranded = stranded
		self.directional = directional
		self.binarize = binarize
		self.anchor = anchor

		#Prepare TFBS in the correct format
		self._prepare_TFBS()
		sites = self._sites #sites still points to ._sites

		#Should strand be taken into account?
		if stranded == True:
			sites = self._sites.copy() #don't change self._sites

			name_to_idx = {}
			TF_names = [] #names in order of idx
			current_idx = 0 #initialize index at 0

			for i, site in enumerate(self.TFBS):
				name = "{0}({1})".format(site.name, site.strand)
				if name not in name_to_idx:
					TF_names.append(name)

					name_to_idx[name] = current_idx
					current_idx += 1 #increment for next name

				sites[i][-1] = name_to_idx[name] #set new idx based on stranded name
		else:
			TF_names = self.TF_names
		self.logger.spam("TF_names: {0}".format(TF_names))

		#Sort sites by mid if anchor is center:
		if anchor == "center": 
			sort_idx = self._get_sort_idx(sites, anchor="center")
			sites = sites[sort_idx, :]

		#---------- Count co-occurrences within TFBS ---------#
		self.logger.info("Counting co-occurrences within sites")
		n_names = len(TF_names)
		anchor_int = anchor_str_to_int[anchor]
		TF_counts, pair_counts = count_co_occurrence(sites, min_dist=min_dist,
															max_dist=max_dist,
															min_overlap=min_overlap,
															max_overlap=max_overlap, 
															binarize=binarize,
															anchor=anchor_int,
															n_names=n_names)
		pair_counts = tfcomb.utils.make_symmetric(pair_counts) if directional == False else pair_counts	#Deal with directionality

		self.count_names = TF_names #this can be different from TF_names if stranded == True
		self.TF_counts = TF_counts
		self.pair_counts = pair_counts

		#---------- Count co-occurrences within shuffled background ---------#
		if n_background > 0:
			self.logger.info("Counting co-occurrence within background")
			parameters = ["min_dist", "max_dist", "min_overlap", "max_overlap", "binarize", "directional"]
			kwargs = {param: getattr(self, param) for param in parameters}
			kwargs["anchor"] = anchor_int
			kwargs["n_names"] = n_names #this is not necessarily the length of self.TF_names!

			#setup multiprocessing
			if threads == 1:
				l = [] #list of background pair_counts

				self.logger.info("Running with multiprocessing threads == 1. To change this, give 'threads' in the parameter of the function.")
				p = Progress(n_background, 10, self.logger) #Setup progress object
				for i in range(n_background):
					l.append(tfcomb.utils.calculate_background(sites, i, **kwargs))
					p.write_progress(i)

			else:
				#Setup pool
				pool = mp.Pool(threads)

				#Add job per background iteration
				jobs = []
				for i in range(n_background):
					self.logger.spam("Adding job for i = {0}".format(i))
					args = (sites, i) #sites, seed
					job = pool.apply_async(tfcomb.utils.calculate_background, args, kwargs) 
					jobs.append(job)
				pool.close()
				
				log_progress(jobs, self.logger) #only exits when the jobs are complete
				
				self.logger.debug("Fetching jobs from pool")
				l = [job.get() for job in jobs]
				pool.join()

			#Calculate z-score per pair
			stacked = np.stack(l) #3-dimensional array of stacked pair_counts from background
			stds = np.std(stacked, axis=0)
			stds[stds == 0] = np.nan #possible divide by zero in zscore calculation
			means = np.mean(stacked, axis=0)
			z = (pair_counts - means)/stds	#pair_counts are the true osberved co-occurrences

			#Handle NaNs introduced in zscore calculation
			z[np.isnan(z) & (pair_counts == means)] = 0
			z[np.isnan(z) & (pair_counts > means)] = np.inf
			z[np.isnan(z) & (pair_counts < means)] = -np.inf
			self.zscore = z

		else:
			self.logger.info("n_background is set to 0; z-score calculation will be skipped")

		self.logger.info("Done finding co-occurrences! Run .market_basket() to estimate significant pairs")

	def get_pair_locations(self, pair, 
								 TF1_strand = None, 
								 TF2_strand = None, 
								 **kwargs):
		""" 
		Get genomic locations of a particular TF pair. Requires .TFBS to be filled. 
		If 'count_within' was run, the parameters used within the latest 'count_within' run are used. Else, the default values of tfcomb.utils.get_pair_locations() are used.
		Both options can be overwritten by setting kwargs.
		
		Parameters
		----------
		pair : tuple
			Name of TF1, TF2 in pair.
		TF1_strand : str, optional
			Strand of TF1 in pair. Default: None (strand is not taken into account).
		TF2_strand : str, optional
			Strand of TF2 in pair. Default: None (strand is not taken into account).
		kwargs : arguments
			Any additional arguments are passed to tfcomb.utils.get_pair_locations.

		Returns
		-------
		tfcomb.utils.TFBSPairList
		"""
		
		#Check input parameters
		self.check_pair(pair)
		self._check_TFBS()
		self._prepare_TFBS() #prepare TFBS sites if not already existing
		self.logger.debug("kwargs given in function: {0}".format(kwargs))

		#If not set, fill in kwargs with internal arguments set by count_within()
		attributes = ["min_dist", "max_dist", "directional", "min_overlap", "max_overlap", "anchor"]
		for att in attributes:
			if hasattr(self, att):
				if not att in kwargs:
					kwargs[att] = getattr(self, att)
		self.logger.debug("kwargs for get_pair_locations: {0}".format(kwargs))	

		TF1, TF2 = pair
		TF1_int = self.name_to_idx[TF1]
		TF2_int = self.name_to_idx[TF2]
		anchor_string = kwargs["anchor"]
		
		#Sort sites based on the anchor position
		sites = self._sites
		if anchor_string == "center":
			sort_idx = self._get_sort_idx(sites, anchor=anchor_string)
			idx_to_original = {idx: original_idx for idx, original_idx in enumerate(sort_idx)} 
			sites = sites[sort_idx, :]

		#Get locations via counting function
		kwargs["anchor"] = self.anchor_to_int[anchor_string] #convert anchor string to int
		idx_mat = tfcomb.counting.count_co_occurrence(sites, task=3, rules=[(TF1_int, TF2_int)], **kwargs)
		n_locations = idx_mat.shape[0]

		#Fetch locations from TFBS list
		locations = tfcomb.utils.TFBSPairList([None]*n_locations)
		if anchor_string == "center":
			for i in range(n_locations):

				site1_idx = idx_mat[i, 0] #location in sorted sites
				site1_idx = idx_to_original[site1_idx] #original idx in self.TFBS (before sorting)

				site2_idx = idx_mat[i, 1]
				site2_idx = idx_to_original[site2_idx]

				#Fetch locations in .TFBS
				site1 = self.TFBS[site1_idx]
				site2 = self.TFBS[site2_idx]
				locations[i] = TFBSPair(TFBS1=site1, TFBS2=site2, anchor=anchor_string)
		else:
			for i in range(n_locations):

				site1_idx = idx_mat[i, 0] #no need to convert idx back to original, since sites were not sorted
				site2_idx = idx_mat[i, 1]

				#Fetch locations in .TFBS
				site1 = self.TFBS[site1_idx]
				site2 = self.TFBS[site2_idx]
				locations[i] = TFBSPair(TFBS1=site1, TFBS2=site2, anchor=anchor_string)

		#Check strandedness
		if TF1_strand != None or TF2_strand != None:
			drop = [] #collect indices to drop from locations
			for i, pair in enumerate(locations):
				
				if TF1_strand != None and ((pair[0].name == TF1 and pair[0].strand != TF1_strand) or (pair[1].name == TF1 and pair[1].strand != TF1_strand)):
					drop.append(i)

				if TF2_strand != None and ((pair[0].name == TF2 and pair[0].strand != TF2_strand) or (pair[1].name == TF2 and pair[1].strand != TF2_strand)):	 
					drop.append(i)

			drop = set(drop) #remove any duplicate indices
			locations = tfcomb.utils.TFBSPairList([pair for i, pair in enumerate(locations) if i not in drop])

		return(locations)

	#-----------------------------------------------------------------------------------------#
	#-------------------------------- Market basket analysis ---------------------------------#
	#-----------------------------------------------------------------------------------------#

	def market_basket(self, measure="cosine", 
							threads=1,
							keep_zero=False,
							n_baskets=1e6,
							_show_columns=["TF1_TF2_count", "TF1_count", "TF2_count"]):
		"""
		Runs market basket analysis on the TF1-TF2 counts. Requires prior run of .count_within().
	
		Parameters
		-----------
		measure : str or list of strings, optional
			The measure(s) to use for market basket analysis. Can be any of: ["cosine", "confidence", "lift", "jaccard"]. Default: 'cosine'.
		threads : int, optional
			Threads to use for multiprocessing. This is passed to .count_within() in case the <CombObj> does not contain any counts yet. Default: 1.
		keep_zero : bool, optional
			Whether to keep rules with 0 occurrences in .rules table. Default: False (remove 0-rules).
		n_baskets : int, optional
			The number of baskets used for calculating market basket measures. Default: 1e6.

		Raises
		-------
		InputError 
			If the measure given is not within available measures.
		"""

		#Check given input
		check_value(threads, vmin=1, name="threads")

		available_measures = ["cosine", "confidence", "lift", "jaccard"]
		if isinstance(measure, str):
			measure = [measure]
		for m in measure:
			tfcomb.utils.check_string(m, available_measures)

		#Check that TF counts are available; otherwise calculate counts
		try:
			self._check_counts()
		except InputError as e:
			print(e)
			self.logger.warning("No counts found in <CombObj>. Running <CombObj>.count_within() with standard parameters.")
			self.count_within(threads=threads)

		#Check show columns; these are the columns which will be shown in .rules output
		available = ["TF1_TF2_count", "TF1_count", "TF2_count", "n_baskets", "TF1_TF2_support", "TF1_support", "TF2_support"]
		for col in _show_columns:
			tfcomb.utils.check_string(col, available)

		##### Calculate market basket analysis #####
	
		#Convert pair counts to table and convert to long format
		pair_counts_table = pd.DataFrame(self.pair_counts, index=self.count_names, columns=self.count_names) #size n x n TFs
		pair_counts_table["TF1"] = pair_counts_table.index
		table = pd.melt(pair_counts_table, id_vars=["TF1"], var_name=["TF2"], value_name="TF1_TF2_count")  #long format (TF1, TF2, value)

		#Add TF single counts to table
		vals = zip(self.count_names, self.TF_counts)
		single_counts = pd.DataFrame(vals, columns=["TF", "count"])
		tf1_counts = single_counts.rename(columns={"TF": "TF1", "count":"TF1_count"})
		tf2_counts = single_counts.rename(columns={"TF": "TF2", "count":"TF2_count"})
		table = table.merge(tf1_counts).merge(tf2_counts)

		#Calculate support
		table["n_baskets"] = n_baskets
		table["TF1_TF2_support"] = table["TF1_TF2_count"] / table["n_baskets"]
		table["TF1_support"] = table["TF1_count"] / table["n_baskets"]
		table["TF2_support"] = table["TF2_count"] / table["n_baskets"]

		#Calculate association metric:
		if isinstance(measure, str):
			measure = [measure]

		for metric in measure:
			if metric == "cosine":
				table["cosine"] = table["TF1_TF2_support"] / np.sqrt(table["TF1_support"] * table["TF2_support"])
			elif metric == "confidence":
				table["confidence"] = table["TF1_TF2_support"] / table["TF1_support"]
			elif metric == "lift":
				table["lift"] = table["confidence"] / table["TF2_support"]
			elif metric == "jaccard":
				table["jaccard"] = table["TF1_TF2_support"] / (table["TF1_support"] + table["TF2_support"] - table["TF1_TF2_support"])
			else:
				raise InputError("Measure '{0}' is invalid. The measure must be one of: {1}".format(metric, available_measures))
		
		#Remove rows with TF1_TF2_count == 0
		if keep_zero == False:
			table = table[table["TF1_TF2_count"] != 0]

		#Sort for highest measure pairs
		table.sort_values([measure[0], "TF1"], ascending=[False, True], inplace=True) #if two pairs have equal measure, sort by TF1 name
		table.reset_index(inplace=True, drop=True)

		#Add z-score per pair
		if hasattr(self, "zscore"):
			zscore_table = pd.DataFrame(self.zscore, index=self.count_names, columns=self.count_names) #size n x n TFs
			zscore_table["TF1"] = zscore_table.index
			ztable_table_long = pd.melt(zscore_table, id_vars=["TF1"], var_name=["TF2"], value_name="zscore")  #long format (TF1, TF2, value)
			table = table.merge(ztable_table_long)
			measure += ["zscore"] #show zscore in output

		#Create internal node table for future network analysis
		TF1_table = table[["TF1", "TF1_count"]].set_index("TF1", drop=False).drop_duplicates()
		TF2_table = table[["TF2", "TF2_count"]].set_index("TF2", drop=False).drop_duplicates()
		self.TF_table = TF1_table.merge(TF2_table, left_index=True, right_index=True)

		#Set name of index for table
		table.index = table["TF1"] + "-" + table["TF2"]

		#Subset to _show_columns
		table = table[["TF1", "TF2"] + _show_columns + measure]

		#Market basket is done; save to .rules
		self.rules = table
		self.logger.info("Market basket analysis is done! Results are found in <CombObj>.rules")
		

	#-----------------------------------------------------------------------------------------#
	#------------------------------ Selecting significant rules ------------------------------#
	#-----------------------------------------------------------------------------------------#

	def reduce_TFBS(self):
		""" Reduce TFBS to the TFs present in .rules.
		
		Returns
		--------
		None - changes .TFBS in place"""

		if hasattr(self, "TFBS"): #This is false for DiffCombObj, which is also using the function

			#Get names from rules
			self.logger.debug("Getting names")
			selected_names = list(set(self.rules["TF1"].tolist() + self.rules["TF2"].tolist()))

			#Do the TFs contain strand information? Collect this for subsetting
			name_strands = {}
			clean_names = []
			for name in selected_names:
				m = re.match("(.+)\(([+-.])\)$", name) #check if name is "NAME(+/./-)"
				if m is not None:
					name_clean = m.group(1) #name without strand
					clean_names.append(name_clean)

					name_strands[name_clean] = name_strands.get(name_clean, set()) | set([m.group(2)]) #add strand to set for this name
				else:
					clean_names.append(name) #name is already clean of strand info

			#Subset TFBS using name and strand information
			self.logger.debug("Setting TFBS in new object")
			clean_names = set(clean_names) #check against set is faster than list
			self.TFBS = RegionList([site for site in self.TFBS if (site.name in clean_names) and (site.strand in name_strands.get(site.name, {site.strand}))]) #comparison for strand is a set
			#if site is not in name_strands, site.strand is compared to itself

			#Update TF names as well (names without strands)
			self.TF_names = [name for name in self.TF_names if name in clean_names]


	def simplify_rules(self):
		""" 
		Simplify rules so that TF1-TF2 and TF2-TF1 pairs only occur once within .rules. 
		This is useful for association metrics such as 'cosine', where the association of TF1->TF2 equals TF2->TF1. 
		This function keeps the first unique pair occurring within the rules table. 
		"""

		self._check_rules()

		#Go through pairs and check which to keep
		tuples = self.rules[["TF1", "TF2"]].to_records(index=False).tolist()
		seen = {}
		idx_to_keep = {}
		for i, tup in enumerate(tuples):
			
			reverse = (tup[1],tup[0])
			if reverse in seen: #the reverse was already found previously
				pass #do not keep
			else:
				seen[tup] = True
				idx_to_keep[i] = True

		#Create simplified table using idx_to_keep
		sub_rules = self.rules.iloc[list(idx_to_keep.keys())]

		self.rules = sub_rules #overwrite .rules with simplified rules
		
	def select_TF_rules(self, TF_list, TF1=True, TF2=True, reduce_TFBS=True, inplace=False, how="inner"):
		""" Select rules based on a list of TF names. The parameters TF1/TF2 can be used to select for which TF to create the selection on (by default: both TF1 and TF2).

		Parameters
		------------
		TF_list : list
			List of TF names fitting to TF1/TF2 within .rules.
		TF1 : bool, optional
			Whether to subset the rules containing 'TF_list' TFs within "TF1". Default: True.
		TF2 : bool, optional
			Whether to subset the rules containing 'TF_list' TFs within "TF2". Default: True.
		reduce_TFBS : bool, optional
			Whether to reduce the .TFBS of the new object to the TFs remaining in `.rules` after selection. Setting this to 'False' will improve speed, but also increase memory consumption. Default: True.
		inplace : bool, optional
			Whether to make selection on current CombObj. If False, 
		how: string, optional
			How to join TF1 and TF2 subset. Default: inner

		Raises
		--------
		InputError
			If both TF1 and TF2 are False or if no rules were selected based on input.

		Returns
		--------
		If inplace == False; tfcomb.objects.CombObj()
			An object containing a subset of <Combobj>.rules.
		if inplace == True; 
			Returns None
		"""

		#Check input
		self._check_rules()
		check_type(TF_list, list, name="TF_list")
		check_type(TF1, bool, "TF1")
		check_type(TF2, bool, "TF2")
		check_string(how, ["left", "right", "outer", "inner", "cross"])

		#Create selected subset
		selected = self.rules

		if TF1 == False and TF2 == False:
			raise InputError("Either TF1 or TF2 must be True in order to create a selection.")

		#Create selections for TF1/TF2
		selections = []
		for (TF_bool, TF_col) in zip([TF1, TF2], ["TF1", "TF2"]):

			if TF_bool == True:

				#Write out any TFs from TF_list not in TF_col names
				not_found = set(TF_list) - set(self.rules[TF_col])
				if len(not_found) > 0:
					self.logger.warning("{0}/{1} names in 'TF_list' were not found within '{2}' names: {3}".format(len(not_found), len(TF_list), TF_col, list(not_found)))

				#Create selection
				selected_bool = self.rules[TF_col].isin(TF_list)
				selected = self.rules[selected_bool]
				selections.append(selected)

		#Join selections from TF1 and TF2
		if len(selections) > 1:
			selected = selections[0].merge(selections[1], how=how) 
		else:
			selected = selections[0]

		#Set index of selected
		selected.index = selected["TF1"] + "-" + selected["TF2"]

		#Stop if no rules were able to be selected
		if len(selected) == 0:
			raise InputError("No rules could be selected - please adjust TF_list and/or TF1/TF2 parameters.")

		self.logger.info("Selected {0} rules".format(len(selected)))

		#Create new object with selected rules (or filter current object)
		if inplace == True:
			self.rules = selected
			self.network = None

			#Reduce the TFBS and TF_names
			if reduce_TFBS == True:
				self.reduce_TFBS()

			return(None)

		else: #create copy of object
			self.logger.info("Creating subset of object")
			new_obj = self.copy()
			new_obj.rules = selected
			new_obj.network = None

			#Reduce the TFBS and TF_names
			if reduce_TFBS == True:
				new_obj.reduce_TFBS()
		
			return(new_obj)
	
	def select_custom_rules(self, custom_list, reduce_TFBS=True):
		""" Select rules based on a custom list of TF pairs. 
		
		Parameters
		------------
		custom_list : list of strings
			List of TF pairs (e.g. a string "TF1-TF2") fitting to TF1/TF2 combination within .rules.
		reduce_TFBS : bool, optional
			Whether to reduce the .TFBS of the new object to the TFs remaining in `.rules` after selection. Setting this to 'False' will improve speed, but also increase memory consumption. Default: True.

		Returns
		--------
		tfcomb.objects.CombObj()
			An object containing a subset of <Combobj>.rules
		"""

		#Check input
		self._check_rules()
		check_type(custom_list, list, name="TF_list")
		#check_type(custom_list[0], [tuple, list], name="Pairs")

		#Create selected subset
		selected = self.rules.copy()

		selected = selected.loc[custom_list]

		#Create new object with selected rules
		new_obj = self.copy()
		new_obj.rules = selected
		new_obj.network = None

		if reduce_TFBS == True:
			new_obj.reduce_TFBS()

		return(new_obj)

	def select_top_rules(self, n, reduce_TFBS=True):
		"""
		Select the top 'n' rules within .rules. By default, the .rules are sorted for the measure value, so n=100 will select the top 100 highest values for the measure (e.g. cosine).

		Parameters
		-----------
		n : int
			The number of rules to select.
		reduce_TFBS : bool, optional
			Whether to reduce the .TFBS of the new object to the TFs remaining in `.rules` after selection. Setting this to 'False' will improve speed, but also increase memory consumption. Default: True.

		Returns
		--------
		tfcomb.objects.CombObj()
			An object containing a subset of <Combobj>.rules
		"""

		#Check input types
		self._check_rules()
		tfcomb.utils.check_type(n, int, "n")

		#Select top n_rules from .rules
		selected = self.rules.copy()
		selected = selected[:n]

		#Create new object with selected rules
		new_obj = self.copy()
		new_obj.rules = selected
		new_obj.network = None

		if reduce_TFBS == True:
			new_obj.reduce_TFBS()
		
		return(new_obj)

	def select_significant_rules(self, x="cosine", 
										y="zscore", 
										x_threshold=None,
										x_threshold_percent=0.05,
										y_threshold=None,
										y_threshold_percent=0.05,
										reduce_TFBS=True,
										plot=True, 
										**kwargs):
		"""
		Make selection of rules based on distribution of x/y-measures

		Parameters
		-----------
		x: str, optional
			The name of the column within .rules containing the measure to be selected on. Default: 'cosine'.
		y : str, optional
			The name of the column within .rules containing the pvalue to be selected on. Default: 'zscore'
		x_threshold : float, optional
			A minimum threshold for the x-axis measure to be selected. If None, the threshold will be estimated from the data. Default: None.
		x_threshold_percent : float between 0-1, optional
			If x_threshold is not given, x_threshold_percent controls the strictness of the automatic threshold selection. Default: 0.05.
		y_threshold : float, optional
			A minimum threshold for the y-axis measure to be selected. If None, the threshold will be estimated from the data. Default: None.
		y_threshold_percent : float between 0-1, optional
			If y_threshold is not given, y_threshold_percent controls the strictness of the automatic threshold selection. Default: 0.05.
		reduce_TFBS : bool, optional
			Whether to reduce the .TFBS of the new object to the TFs remaining in `.rules` after selection. Setting this to 'False' will improve speed, but also increase memory consumption. Default: True.
		plot : bool, optional
			Whether to show the 'measure vs. pvalue'-plot or not. Default: True.
		kwargs : arguments
			Additional arguments are forwarded to tfcomb.plotting.scatter

		Returns
		--------
		tfcomb.objects.CombObj()
			An object containing a subset of <obj>.rules

		See also
		---------
		tfcomb.plotting.scatter
		"""

		#Check given input
		self._check_rules()
		if x_threshold is not None:
			check_value(x_threshold)
		if y_threshold is not None:
			check_value(y_threshold)
		tfcomb.utils.check_value(y_threshold_percent, vmin=0, vmax=1, name="y_threshold_percent")
		tfcomb.utils.check_value(x_threshold_percent, vmin=0, vmax=1, name="x_threshold_percent")	

		#Check if measure are in columns
		if x not in self.rules.columns:
			raise KeyError("Column given for x ('{0}') is not in .rules".format(x))

		#Warn if y aka zscore contains inf
		if self.rules[y].isin([np.inf, -np.inf]).any():
			self.logger.warning(f"{y} contains infinite value. This could be the result of 'n_background' from .count_within being set too low, which will lead to spurious results of .select_significant_rules. Please check and adjust parameters if necessary.")

		#If measure_threshold is None; try to calculate optimal threshold via knee-plot
		if x_threshold is None:
			self.logger.info("x_threshold is None; trying to calculate optimal threshold")
			x_threshold = tfcomb.utils.get_threshold(self.rules[x], percent=x_threshold_percent)

		if y_threshold is None:
			self.logger.info("y_threshold is None; trying to calculate optimal threshold")
			y_threshold = tfcomb.utils.get_threshold(self.rules[y], percent=y_threshold_percent)

		#Set threshold on table
		selected = self.rules.copy()
		selected = selected[(selected[x] >= x_threshold) & (selected[y] >= y_threshold)]

		if plot == True:
			tfcomb.plotting.scatter(self.rules, x=x, 
												y=y, 
												x_threshold=x_threshold,
												y_threshold=y_threshold,
												**kwargs)

		#Create a CombObj with the subset of TFBS and rules
		self.logger.info("Creating subset of TFBS and rules using thresholds")
		self.logger.debug("Copying old to new object")
		new_obj = self.copy()
		new_obj.rules = selected
		new_obj.network = None

		if reduce_TFBS == True:
			new_obj.reduce_TFBS()	

		return(new_obj)

	#-----------------------------------------------------------------------------------------#
	#------------------------------ Integration of external data -----------------------------#
	#-----------------------------------------------------------------------------------------#

	def integrate_data(self, table, merge="pair", TF1_col="TF1", TF2_col="TF2", prefix=None):
		""" Function to add external data to object rules.
		
		Parameters
		------------
		table : str or pandas.DataFrame
			A table containing data to add to .rules. If table is a string, 'table' is assumed to be the path to a tab-separated table containing a header line and rows of data.
		merge : str
			Which information to merge - must be one of "pair", "TF1" or "TF2". The option "pair" is used to merge infromation about TF-TF pairs such as protein-protein-interactions.
			The 'TF1' and 'TF2' can be used to include TF-specific information such as expression levels.
		TF1_col : str, optional
			The column in table corresponding to "TF1" name. If merge == "TF2", 'TF1' is ignored. Default: "TF1".
		TF2_col : str, optional
			The column in table corresponding to "TF2" name. If merge == "TF1", 'TF2' is ignored. Default: "TF2".
		prefix : str, optional
			A prefix to add to the columns. Can be useful for adding the same information to both TF1 and TF2 (e.g. by using "TF1" and "TF2" prefixes),
			or adding same-name columns from different tables. Default: None (no prefix).
		"""

		self._check_rules()
		check_type(table, [str, pd.DataFrame], "table")
		check_string(merge, ["pair", "TF1", "TF2"], "merge")

		#Read table if string (path) was given
		if isinstance(table, str):
			table = pd.read_csv(table, sep="\t")
			self.logger.info("Read table of shape {0} with columns: {1}".format(table.shape, table.columns.tolist()))
		
		table = table.drop_duplicates()

		#Add prefix to columns
		if prefix is not None:
			check_type(prefix, str)
			table.columns = [prefix + str(col) if col not in [TF1_col, TF2_col] else col for col in table.columns]

		#Check if columns in table were already existing
		current_columns = [col for col in self.rules.columns if col not in ["TF1", "TF2"]]
		adding_columns = [col for col in table.columns if col not in [TF1_col, TF2_col]]
		duplicates = list(set(current_columns) & set(adding_columns))
		if len(duplicates) > 1:
			self.logger.warning("Column(s) '{0}' from input table are already present in .rules, and could not be integrated.".format(duplicates))
			self.logger.warning("Please set 'prefix' in order to make the column names unique.")
			table.drop(columns=duplicates, inplace=True)

		#Merge table to object
		if merge == "TF1":
			check_columns(table, [TF1_col])
			self.rules = self.rules.merge(table, left_on="TF1", right_on=TF1_col, how="left")
			self.rules = self.rules.drop(columns=[TF1_col])
		
		elif merge == "TF2":
			check_columns(table, [TF2_col])
			self.rules = self.rules.merge(table, left_on="TF2", right_on=TF2_col, how="left")
			self.rules = self.rules.drop(columns=[TF2_col])

		elif merge == "pair":
			check_columns(table, [TF1_col, TF2_col])
			self.rules = self.rules.merge(table, left_on=["TF1", "TF2"], right_on=[TF1_col, TF2_col], how="left")
		
		#Set name of index for table
		self.rules.index = self.rules["TF1"] + "-" + self.rules["TF2"]

		#If data was integrated, .network must be recalculated	
		self.network = None
		

	#-----------------------------------------------------------------------------------------#
	#-------------------------------- Plotting functionality  --------------------------------#
	#-----------------------------------------------------------------------------------------#

	def plot_TFBS(self, **kwargs):
		"""
		This is a wrapper for the plotting function `tfcomb.plotting.genome_view`

		Parameters
		------------
		kwargs : arguments
			All arguments are passed to `tfcomb.plotting.genome_view`. Please see the documentation for input parameters.
		
		"""

		self._check_TFBS() #Requires TFBS

		#Plot TFBS via genome view
		tfcomb.plotting.genome_view(self.TFBS, **kwargs)


	def plot_heatmap(self, n_rules=20, color_by="cosine", sort_by=None, **kwargs):
		"""
		Plot a heatmap of rules and their attribute values. This is a wrapper for the plotting function `tfcomb.plotting.heatmap`.

		Parameters
		-----------
		n_rules : int, optional
			The number of rules to show. The first `n_rules` rules of .rules are taken. Default: 20.
		color_by : str, optional
			A column within .rules to color the heatmap by. Note: Can be different than sort_by. Default: "cosine".
		sort_by : str, optional
			A column within .rules to sort by before choosing n_rules. Default: None (rules are not sorted before selection).
		kwargs : arguments
			Any additional arguments are passed to tfcomb.plotting.heatmap.

		See also
		---------
		tfcomb.plotting.heatmap
		"""

		#Check types
		tfcomb.utils.check_type(n_rules, [int])

		#Check that columns are available in self.rules
		tfcomb.utils.check_columns(self.rules, [color_by, sort_by])
				
		#Sort table by another column than currently
		associations = self.rules.copy()
		if sort_by is not None:
			associations = associations.sort_values(sort_by, ascending=False)

		#Choose n number of rules
		tf1_list = associations["TF1"][:n_rules]
		tf2_list = associations["TF2"][:n_rules]

		# Fill all combinations for the TFs selected from top rules (to fill white spaces)
		chosen_associations = associations[(associations["TF1"].isin(tf1_list) &
											associations["TF2"].isin(tf2_list))]

		#Plot
		h = tfcomb.plotting.heatmap(chosen_associations, color_by=color_by, **kwargs)


	def plot_bubble(self, n_rules=20, yaxis="cosine", color_by="TF1_TF2_count", size_by=None, sort_by=None, **kwargs):
		"""
		Plot a bubble-style scatterplot of the object rules. This is a wrapper for the plotting function `tfcomb.plotting.bubble`.
		
		Parameters
		-----------
		n_rules : int, optional
			The number of rules to show. The first `n_rules` rules of .rules are taken. Default: 20.
		yaxis : str, optional
			A column within .rules to depict on the y-axis of the plot. Default: "cosine".	
		color_by : str, optional
			A column within .rules to color points in the plot by. Default: "TF1_TF2_count".
		size_by : str, optional
			A column within .rules to size points in the plot by. Default: None.
		sort_by : str, optional
			A column within .rules to sort by before choosing n_rules. Default: None (rules are not sorted before selection).
		unique : bool, optional
			Only show unique pairs in plot, e.g. only the first occurrence of TF1-TF2 / TF2-TF1. Default: True.
		kwargs : arguments
			Any additional arguments are passed to tfcomb.plotting.bubble.

		See also
		-----------
		tfcomb.plotting.bubble
		"""

		self._check_rules()

		#Sort rules
		table = self.rules.copy()
		if sort_by is not None:
			table = table.sort_values(sort_by, ascending=False)

		#Select n top rules
		top_rules = table.head(n_rules)
		top_rules.index = top_rules["TF1"].values + " + " + top_rules["TF2"].values

		#Plot
		ax = tfcomb.plotting.bubble(top_rules, yaxis=yaxis, color_by=color_by, size_by=size_by, **kwargs)


	def plot_scatter(self, x, y, hue=None, **kwargs):
		""" Plot a scatterplot of information from .rules.
		
		Parameters
		------------
		x : str
			The name of the column in .rules containing values to plot on x-axis.
		y : str
			The name of the column in .rules containing values to plot on y-axis.
		"""
		
		#Check that market basket was run
		self._check_rules()

		tfcomb.plotting.scatter(self.rules, x, y)


	#-------------------------------------------------------------------------------------------#
	#----------------------------------- In-depth analysis -------------------------------------#
	#-------------------------------------------------------------------------------------------#

	def create_distObj(self):
		""" Creates a distObject, useful for manual analysis. 
			 Fills self.distObj.
		"""
		self._check_rules()
		
		self.distObj = DistObj()
		self.distObj.fill_rules(self)
		self.distObj.logger.info("DistObject successfully created! It can be accessed via <CombObj>.distObj")

	def analyze_distances(self, parent_directory=None, threads=4, correction=True, scale=True, **kwargs):
		""" Standard distance analysis workflow.
			Use create_distObj for own workflow steps and more options!
		"""
		self.create_distObj()
		self.distObj.set_verbosity(self.verbosity)
		self.distObj.count_distances(directional=self.directional)

		#Ensure that parent directories exist before trying to plot
		tfcomb.utils.check_type(parent_directory,[type(None),str])
		if parent_directory is not None:
			tfcomb.utils.check_dir(parent_directory)
			subfolder_linres = os.path.join(parent_directory, "linres")
			tfcomb.utils.check_dir(subfolder_linres) 
			subfolder_corrected = os.path.join(parent_directory, "corrected")
			tfcomb.utils.check_dir(subfolder_corrected) 
			subfolder_peaks = os.path.join(parent_directory, "peaks","peaks.tsv")
			tfcomb.utils.check_dir(subfolder_peaks)
		else:
			subfolder_linres = parent_directory
			subfolder_corrected = parent_directory
			subfolder_peaks = parent_directory
		
		#Perform steps in standard workflow
		if scale:
			self.distObj.scale()

		self.distObj.smooth(window_size=3)
		if correction:
			self.distObj.correct_background(threads=threads)
		self.distObj.analyze_signal_all(threads=threads, save=subfolder_peaks, **kwargs)

		if parent_directory is not None:
			for tf1,tf2 in list(zip(self.distances.TF1, self.distances.TF2)):
				self.plot_analyzed_signal((tf1, tf2), only_peaking=True, save=os.path.join(parent_directory, "peaks", f"{tf1}_{tf2}.png"))
		

	def analyze_orientation(self):
		""" Analyze preferred orientation of sites in .TFBS. This is a wrapper for tfcomb.analysis.orientation().
		
		Returns 
		-----------
		pd.DataFrame
		
		See also
		----------
		tfcomb.analysis.orientation
		"""

		self._check_rules() #market basket must be run.

		table = tfcomb.analysis.orientation(self.rules) 

		return(table)

	#-------------------------------------------------------------------------------------------#
	#------------------------------------ Network analysis -------------------------------------#
	#-------------------------------------------------------------------------------------------#

	def build_network(self, **kwargs):
		""" 
		Builds a TF-TF co-occurrence network for the rules within object. This is a wrapper for the tfcomb.network.build_nx_network() function, 
		which uses the python networkx package. 
		
		Parameters
		------------
		kwargs : arguments
			Any additional arguments are passed to tfcomb.network.build_nx_network().

		Returns
		-------
		None - fills the .network attribute of the `CombObj` with a networkx.Graph object
		"""

		#Build network
		self.logger.debug("Building network using tfcomb.network.build_network")
		self.network = tfcomb.network.build_network(self.rules, node_table=self.TF_table, verbosity=self.verbosity, **kwargs)
		self.logger.info("Finished! The network is found within <CombObj>.network.")
	

	def cluster_network(self, method="louvain", weight=None):
		"""
		Creates a clustering of nodes within network and add a new node attribute "cluster" to the network. 
		
		Parameters
		-----------
		method : str, one of ["louvain", "blockmodel"]
			The method Default: "louvain".
		weight : str, optional
			The name of the edge attribute to use as weight. Default: None (not weighted).
		"""

		#Fetch network from object
		if self.network is None:
			self.logger.info("The .network attribute is not available - running .build_network()")
			self.build_network()
			
		#Decide method of partitioning
		if method == "louvain":
			tfcomb.network.cluster_louvain(self.network, weight=weight, logger=self.logger) #this adds "partition" to the network
			self.logger.info("Added 'cluster' attribute to the network attributes")

			node_table = tfcomb.network.get_node_table(self.network)

		elif method == "blockmodel":

			#Create gt network	
			self._gt_network = tfcomb.network.build_network(self.rules, node_table=self.TF_table, tool="graph-tool", verbosity=self.verbosity)

			#Partition network
			tfcomb.network.cluster_blockmodel(self._gt_network)

			node_table = tfcomb.network.get_node_table(self._gt_network)
			node_table.set_index("TF1", drop=False, inplace=True)

		else:
			raise ValueError("Method must be one of: ['louvain', 'blockmodel']")
			
		#Update TF_table
		self.logger.debug("TF_table: {0}".format(node_table.head(5)))
		self.TF_table = node_table
		
		#Update network attribute for plotting
		if method == "blockmodel":
			self.network = tfcomb.network.build_network(self.rules, node_table=self.TF_table, verbosity=self.verbosity)

		#no return - networks were changed in place

	def plot_network(self, color_node_by="TF1_count",
						   color_edge_by="cosine", 
						   size_edge_by="TF1_TF2_count",
						   **kwargs): 
		"""
		Plot the rules in .rules as a network using Graphviz for python. This function is a wrapper for 
		building the network (using tfcomb.network.build_network) and subsequently plotting the network (using tfcomb.plotting.network).

		Parameters
		-----------
		color_node_by : str, optional
			A column in .rules or .TF_table to color nodes by. Default: 'TF1_count'.
		color_edge_by : str, optional
			A column in .rules to color edges by. Default: 'cosine'.
		size_edge_by : str, optional
			A column in rules to size edge width by. Default: 'TF1_TF2_count'.
		kwargs : arguments
			All other arguments are passed to tfcomb.plotting.network.

		See also
		--------
		tfcomb.network.build_network and tfcomb.plotting.network
		"""

		#Fetch network from object or build network
		if self.network is None:
			self.logger.warning("The .network attribute is not set yet - running build_network().")
			self.build_network()			#running build network()
			
		#Plot network
		G = self.network 
		dot = tfcomb.plotting.network(G, color_node_by=color_node_by, 
										 color_edge_by=color_edge_by, 
										 size_edge_by=size_edge_by, 
										 verbosity=self.verbosity, **kwargs)

		return(dot)


	#-----------------------------------------------------------------------------------------#
	#------------------------------ Comparison to other objects ------------------------------#
	#-----------------------------------------------------------------------------------------#

	def compare(self, obj_to_compare, measure="cosine", join="inner", normalize=True):
		"""
		Utility function to create a DiffCombObj directly from a comparison between this CombObj and another CombObj. Requires .market_basket() run on both objects.
		Runs DiffCombObj.normalize (if chosen) and DiffCombObj.calculate_foldchanges() under the hood. 

		Note
		------
		Set .prefix for each object to get proper naming of output log2fc columns. 

		Parameters
		---------
		obj_to_compare : tfcomb.objects.CombObj
			Another CombObj to compare to the current CombObj.
		measure : str, optional
			The measure to compare between objects. Default: 'cosine'.
		join : string
			How to join the TF names of the two objects. Must be one of "inner" or "outer". If "inner", only TFs present in both objects are retained. 
			If "outer", TFs from both objects are used, and any missing counts are set to 0. Default: "inner".
		normalize : bool, optional
			Whether to normalize values between objects. Default: True.

		Return
		-------
		DiffCombObj
		"""
		
		#Create object
		diff = DiffCombObj([self, obj_to_compare], measure=measure, join=join, verbosity=self.verbosity)

		if normalize == True:
			diff.normalize()

		diff.calculate_foldchanges()

		return(diff)



###################################################################################
############################## Differential analysis ##############################
###################################################################################


class DiffCombObj():

	def __init__(self, objects=[], measure='cosine', join="inner", fillna=True, verbosity=1):
		""" Initializes a DiffCombObj object for doing differential analysis between CombObj's.

		Parameters
		------------
		objects : list, optional
			A list of CombObj instances. If list is empty, an empty DiffCombObj will be created. Default: [].
		measure : str, optional
			The measure to compare between objects. Must be a column within .rules for each object. Default: 'cosine'.
		join : string
			How to join the TF names of the two objects. Must be one of "inner" or "outer". If "inner", only TFs present in both objects are retained. 
			If "outer", TFs from both objects are used, and any missing counts are set to 0. Default: "inner".
		fillna : True
			If "join" == "outer", there can be missing counts for individual rules. If fillna == True, these counts are set to 0. Else, the counts are NA. 
			Default: True.
		verbosity : int, optional
			The verbosity of the output logging. Default: 1.

		See also
		---------
		add_object for adding objects one-by-one

		"""
		
		#Initialize object variables
		self.n_objects = 0
		self.prefixes = [] #filled by ".add_object"
		self.measure = measure	#the measure

		#Setup logger
		self.verbosity = verbosity
		self.logger = TFcombLogger(self.verbosity)

		#Add objects one-by-one
		for obj in objects:
			self.add_object(obj, join=join, fillna=fillna)

		#Use functions from CombObj
		self._set_combobj_functions()

	def _set_combobj_functions(self):
		""" Reuse CombObj functions"""

		self.copy = lambda : CombObj.copy(self)
		self.set_verbosity = lambda *args, **kwargs: CombObj.set_verbosity(self, *args, **kwargs)
		self.build_network = lambda : CombObj.build_network(self)
		self._check_rules = lambda : CombObj._check_rules(self)
		self.simplify_rules = lambda : CombObj.simplify_rules(self)
		self.select_TF_rules = lambda *args, **kwargs: CombObj.select_TF_rules(self, *args, **kwargs)
		self.select_custom_rules = lambda  *args, **kwargs: CombObj.select_custom_rules(self, *args, **kwargs)
		self.reduce_TFBS = lambda : CombObj.reduce_TFBS(self) #to use in selecting rules
		self.integrate_data = lambda *args, **kwargs: CombObj.integrate_data(self, *args, **kwargs)

	def __str__(self):
		pass
		
	def add_object(self, obj, join="inner", 
							 fillna=True
							  ):
		"""
		Add one CombObj to the DiffCombObj.

		Parameters
		-----------
		obj : CombObj
			An instance of CombObj
		join : string
			How to join the TF names of the two objects. Must be one of "inner" or "outer". If "inner", only TFs present in both objects are retained. 
			If "outer", TFs from both objects are used, and any missing counts are set to 0. Default: "inner".
		fillna : True
			If "join" == "outer", there can be missing counts for individual rules. If fillna == True, these counts are set to 0. Else, the counts are NA. 
			Default: True.

		Returns
		--------
		None
			Object is added in place
		"""

		#Check that object is an instance of CombObj
		check_type(obj, [CombObj])
		check_string(join, ["inner", "outer"], "join")

		#Check that market basket was run on the object
		try:
			obj._check_rules()
		except InputError as e:
			raise InputError("Object is missing .rules. Please check that .market_basket() was run on the CombObj.")

		#Check if prefix is set - otherwise, set to obj<int>
		if obj.prefix is not None:
			prefix = obj.prefix
		else:
			prefix = "Obj" + str(self.n_objects + 1)
			self.logger.warning("CombObj has no prefix set, so the prefix in the DiffCombObj was set to '{0}'. Use <CombObj>.set_prefix() to set a specific prefix for the CombObj.".format(prefix))
		self.prefixes.append(prefix)

		#Check that prefixes are unique (e.g. if one prefix was set to "Obj1")
		duplicates = set([p for p in self.prefixes if self.prefixes.count(p) > 1])
		if len(duplicates) > 0:
			raise InputError("Prefix {0} is not unique within DiffCombObj prefixes. Please use <CombObj>.set_prefix() to set another prefix for the object. The current prefixes are: {1}".format(prefix, self.prefixes))
		
		#check that object contains self.measure
		if self.measure not in obj.rules.columns:
			raise InputError("Measure '{0}' is not available in <CombObj>.rules. Please rerun .market_basket() for this measure or select another measure for the DiffCombObj".format(self.measure))

		#Format table from obj to contain TF1/TF2 + measures with prefix
		columns_to_keep = ["TF1", "TF2"] + [self.measure]
		
		obj_table = obj.rules[columns_to_keep] #only keep necessary columns
		obj_table.columns = [str(prefix) + "_" + col if col not in ["TF1", "TF2"] else col for col in obj_table.columns] #add prefix to all columns besides TF1/TF2

		obj_TF_table = obj.TF_table.copy()
		obj_TF_table.columns = [str(prefix) + "_" + col for col in obj_TF_table.columns]

		#Initialize tables if this is the first object
		if self.n_objects == 0: 
			self.rules = obj_table
			self.TF_table = obj_TF_table

		#Or add object to this DiffCombObj
		else:
			
			#if join is inner, remove any TFs not present in both objects
			if join == "inner":

				left_TFs = set(list(set(self.rules["TF1"])) + list(set(self.rules["TF2"])))
				right_TFs = set(list(set(obj_table["TF1"])) + list(set(obj_table["TF2"])))

				common = left_TFs.intersection(right_TFs)
				not_common = left_TFs.union(right_TFs) - common
				if len(not_common) > 0:
					self.logger.warning("{0} TFs were not common between objects and were excluded from .rules. Set 'join' to 'outer' in order to use all TFs across objects. The TFs excluded were: {1}".format(len(not_common), list(not_common)))

				#Subset both tables to common TFs
				A = self.rules.loc[self.rules["TF1"].isin(common) & self.rules["TF2"].isin(common)]
				B = obj_table.loc[obj_table["TF1"].isin(common) & obj_table["TF2"].isin(common)]

				self.rules = A.merge(B, left_on=["TF1", "TF2"], right_on=["TF1", "TF2"])

			else: #join is outer
				self.rules = self.rules.merge(obj_table, left_on=["TF1", "TF2"], right_on=["TF1", "TF2"], how="outer")

				if fillna == True:
					self.rules = self.rules.fillna(0) #Fill NA with null (happens if TF1/TF2 pairs are different between objects)

			#Merge TF tables
			self.TF_table = self.TF_table.merge(obj_TF_table, left_index=True, right_index=True)

		self.n_objects += 1 #current number of objects +1 for the one just added
	
		#Set name of index for table
		self.rules.index = self.rules["TF1"] + "-" + self.rules["TF2"]


	#-----------------------------------------------------------------------------------------#
	#--------------------------- Calculate differential measures -----------------------------#
	#-----------------------------------------------------------------------------------------#

	def normalize(self):
		"""
		Normalize the values for the DiffCombObj given measure (.measure) using quantile normalization. 
		Overwrites the <prefix>_<measure> columns in .rules with the normalized values.
		"""

		#Establish input/output columns
		measure_columns = [prefix + "_" + self.measure for prefix in self.prefixes]
		zero_bool = self.rules[measure_columns] == 0
		nan_bool = self.rules[measure_columns].isnull()

		#Fill na with 0
		data = self.rules[measure_columns]
		data[nan_bool] = 0

		#Normalize values
		self.rules[measure_columns] = qnorm.quantile_normalize(data, axis=1)
		
		#Ensure that original 0 values are kept at 0, and original nan kept at nan
		self.rules[zero_bool] = np.nan
		self.rules.fillna(0, inplace=True)
		self.rules[nan_bool] = np.nan

	def calculate_foldchanges(self, pseudo=0.01):
		""" Calculate measure foldchanges  between objects in DiffCombObj. The measure is chosen at the creation of the DiffCombObj and defaults to 'cosine'.
		
		Parameters
		----------
		pseudo : float, optional
			Set the pseudocount to add to all values before log2-foldchange transformation. Default: 0.01.
	
		See also
		--------
		tfcomb.DiffCombObj.normalize
		"""

		measure = self.measure

		#Find all possible combinations of objects
		combinations = itertools.combinations(self.prefixes, 2)
		self.contrasts = list(combinations)
		self.logger.debug("Contrasts: {0}".format(self.contrasts))

		columns = [] #collect the log2fc columns per contrast
		for (p1, p2) in self.contrasts:
			self.logger.info("Calculating foldchange for contrast: {0} / {1}".format(p1, p2))
			log2_col = "{0}/{1}_{2}_log2fc".format(p1, p2, measure)
			columns.append(log2_col)

			p1_values = self.rules[p1 + "_" + measure]
			p2_values = self.rules[p2 + "_" + measure]

			ratio = (p1_values + pseudo) / (p2_values + pseudo)
			ratio[ratio <= 0] = np.nan 
			self.rules[log2_col] = np.log2(ratio)

		#Sort by first contrast log2fc
		self.logger.debug("columns: {0}".format(columns))
		self.rules.sort_values(columns[0], inplace=True)

		self.logger.info("The calculated log2fc's are found in the rules table (<DiffCombObj>.rules)")
		

	#-----------------------------------------------------------------------------------------#
	#------------------------------ Selecting significant rules ------------------------------#
	#-----------------------------------------------------------------------------------------#

	def select_rules(self, contrast=None,
						   measure="cosine", 
						   measure_threshold=None,
						   measure_threshold_percent=0.05,
						   mean_threshold=None,
						   mean_threshold_percent=0.05,
						   plot = True, 
						   **kwargs):
		"""
		Select differentially regulated rules using a MA-plot on the basis of measure and mean of measures per contrast.
		
		Parameters
		-----------
		contrast : tuple
			Name of the contrast to use in tuple format e.g. (<prefix1>,<prefix2>). Default: None (the first contrast is shown).
		measure : str, optional
			The measure to use for selecting rules. Default: "cosine" (internally converted to <prefix1>/<prefix2>_<measure>_log2fc).
		measure_threshold : tuple, optional
			Threshold for 'measure' for selecting rules. Default: None (the threshold is estimated automatically) 
		measure_threshold_percent : float between 0-1
			If measure_threshold is not set, measure_threshold_percent controls the strictness of the automatic threshold. If you increase this value, more differential rules will be found and vice versa. Default: 0.05.  
		mean_threshold : float, optional
			Threshold for 'mean' for selecting rules. Default: None (the threshold is estimated automatically) 
		mean_threshold_percent : float between 0-1
			if mean_threshold is not set, mean_threshold_percent controls the strictness of the automatic threshold. If you increase this value, more differential rules will be found and vice versa. Default: 0.05.  
		plot : boolean, optional
			Whether to plot the volcano plot. Default: True.
		kwargs : arguments, optional
			Additional arguments are passed to tfcomb.plotting.scatter.

		Returns
		----------
		tfcomb.objects.DiffCombObj()
			An object containing a subset of <DiffCombobj>.rules

		See also
		----------
		tfcomb.plotting.volcano
		"""
		
		table = self.rules.copy() #make sure not to change self.rules

		if self.contrasts is None:
			self.logger.warning(".select_rules requires foldchanges to run. Running .calculate_foldchanges() now.")

		#Check input
		if measure_threshold is not None:
			tfcomb.utils.check_type(measure_threshold, [tuple, list], name="measure_threshold")
			measure_threshold = tuple(sorted(measure_threshold)) #ensure that lower threshold is first in tuple
		if mean_threshold is not None:
			tfcomb.utils.check_value(mean_threshold, name="mean_threshold")

		tfcomb.utils.check_value(measure_threshold_percent, vmin=0, vmax=1, name="measure_threshold_percent")
		tfcomb.utils.check_value(mean_threshold_percent, vmin=0, vmax=1, name="mean_threshold_percent")

		#Identify measure to use based on contrast
		if contrast == None:
			contrast = self.contrasts[0]
		else:
			#check if contrast is valid
			if contrast not in self.contrasts:
				raise InputError("Given contrast {0} is not valid. The contrast must be a tuple and be any of: {1}".format(contrast, self.contrasts))
		
		self.logger.info("Selecting rules for contrast: {0}".format(contrast))
		measure_col = "{0}/{1}_{2}_log2fc".format(contrast[0], contrast[1], measure)
		self.logger.debug("Measure column is: {0}".format(measure_col))

		#Calculate mean of measures
		measure_cols = [c + "_" + measure for c in contrast]
		mean_col = "Mean of '{0}' for {1} and {2}".format(measure, contrast[0], contrast[1])
		table[mean_col] = table[measure_cols].mean(axis=1)
		to_keep = table[mean_col] > 0 #only keep values with mean measure > 0

		#Find optimal measure threshold
		if measure_threshold is None:
			self.logger.info("measure_threshold is None; trying to calculate optimal threshold")
			vals = table[measure_col][to_keep]
			measure_threshold = tfcomb.utils.get_threshold(vals, "both", percent=measure_threshold_percent, verbosity=self.verbosity)
			self.logger.debug("Measure threshold is: {0}".format(measure_threshold))

		#Find optimal mean threshold
		if mean_threshold is None:
			self.logger.info("mean_threshold is None; trying to calculate optimal threshold")
			vals = table[mean_col][to_keep] #remove large influence of 0-means
			mean_threshold = tfcomb.utils.get_threshold(vals, "upper", percent=mean_threshold_percent, verbosity=self.verbosity)
			self.logger.debug("Mean threshold is: {0}".format(mean_threshold))

		#Plot the MA-plot if chosen
		if plot == True:
	
			tfcomb.plotting.scatter(table, 
									x=mean_col, 
									y=measure_col, 
									x_threshold=mean_threshold,
									y_threshold=measure_threshold,
									**kwargs)

		#Set threshold on rules
		selected = self.rules.copy()
		selected = selected[((selected[measure_col] <= measure_threshold[0]) | (selected[measure_col] >= measure_threshold[1])) &
							(table[mean_col] >= mean_threshold)]

		#Create a DiffCombObj with the subset of  rules
		self.logger.info("Creating subset of rules using thresholds")
		new_obj = self.copy()
		new_obj._set_combobj_functions() #set combobj functions for new object; else they point to self
		new_obj.rules = selected
		new_obj.network = None
		
		return(new_obj)



	#-------------------------------------------------------------------------------------------#
	#----------------------------- Plots for differential analysis -----------------------------#
	#-------------------------------------------------------------------------------------------#

	def plot_correlation(self, method="pearson", save=None, **kwargs):
		"""
		Plot correlation of 'measure' between rules across objects.

		Parameters
		-----------
		method : str, optional
			Either 'pearson' or 'spearman'. Default: 'pearson'.
		save : str, optional
			Save the plot to the file given in 'save'. Default: None.
		kwargs : arguments, optional
			Additional arguments are passed to sns.clustermap.
		"""

		#Define columns
		cols = [prefix + "_" + self.measure for prefix in self.prefixes]

		#Calculate matrix and plot
		matrix = self.rules[cols].corr(method=method)

		g = sns.clustermap(matrix,
							cbar_kws={'label': method.capitalize() + " corr."}, **kwargs)

		#rotate x-axis labels
		_ = plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, size=15)  # For y axis
		_ = plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", size=15) # For x axis

		if save is not None:
			plt.savefig(save, dpi=600, bbox_inches="tight")

		return(g)

	def plot_rules_heatmap(self, **kwargs):
		""" Plot a heatmap of size n_rules x n_objects """ 

		cols = [prefix + "_" + self.measure for prefix in self.prefixes]

		data = self.rules[cols]
		g = sns.clustermap(data, xticklabels=True, **kwargs)

		#Rotate labels
		_ = plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", size=15) # For x axis

		return(g)

	#todo: plot_contrast heatmap
	def plot_heatmap(self, contrast=None, 
						   n_rules=10, 
						   color_by="cosine_log2fc", 
						   sort_by=None, 
						   **kwargs):
		"""
		Functionality to plot a heatmap of differentially co-occurring TF pairs for a certain contrast. 

		Parameters
		------------
		contrast : tuple, optional
			Name of the contrast to use in tuple format e.g. (<prefix1>,<prefix2>). Default: None (the first contrast is shown).
		n_rules : int, optional
			Number of rules to show from each contrast (default: 10). Note: This is the number of rules either up/down, meaning that the rules shown are n_rules * 2.
		color_by : str, optional
			Default: "cosine" (converted to "<prefix1>/<prefix2>_<color_by>")
		sort_by : str, optional
			Column in .rules to sort rules by. Default: None (keep sort)
		kwargs : arguments, optional
			Additional arguments are passed to tfcomb.plotting.heatmap.

		See also
		----------
		tfcomb.plotting.heatmap
		"""

		#todo: requires log2fcs to be calculated
		contrast = tfcomb.utils.set_contrast(contrast, self.contrasts)

		#Decide columns based on color_by / sort_by
		color_by = "{0}/{1}_{2}".format(contrast[0], contrast[1], color_by)
		if sort_by is not None:
			sort_by = "{0}/{1}_{2}".format(contrast[0], contrast[1], sort_by)

		#Check if columns are found in table
		check_columns(self.rules, [sort_by, color_by])

		#Sort by measure
		associations = self.rules.copy()
		if sort_by is not None:
			associations.sort_values(sort_by, ascending=False, inplace=True)

		#Choose n number of rules
		tf1_list = list(set(associations["TF1"][:n_rules].tolist() + associations["TF1"][-n_rules:].tolist()))
		tf2_list = list(set(associations["TF2"][:n_rules].tolist() + associations["TF2"][-n_rules:].tolist()))

		# Fill all combinations for the TFs selected from top rules (to fill white spaces)
		chosen_associations = associations[(associations["TF1"].isin(tf1_list) &
											associations["TF2"].isin(tf2_list))]

		#Plot heatmap
		tfcomb.plotting.heatmap(chosen_associations, color_by=color_by)

	def plot_bubble(self, contrast=None,
						  n_rules=20, 
						  yaxis="cosine_log2fc",
						  color_by=None, 
						  size_by=None, 
						  **kwargs):
		"""
		Plot bubble scatterplot of information within .rules.

		Parameters
		-----------
		contrast : tuple, optional
			Name of the contrast to use in tuple format e.g. (<prefix1>,<prefix2>). Default: None (the first contrast is shown).
		n_rules : int, optional
			Number of rules to show (in each direction). Default: 20.
		yaxis : str, optional
			Measure to show on the y-axis. Default: "cosine_log2fc".
		color_by : str, optional
			If column is not in rules, the string is supposed to be in the form "prefix1/prefix2_<color_by>". Default: None.
		size_by : str, optional
			Column to size bubbles by. Default: None.
		kwargs : arguments
			Any additional arguments are passed to tfcomb.plotting.bubble.

		See also
		----------
		tfcomb.plotting.bubble
		"""

		contrast = tfcomb.utils.set_contrast(contrast, self.contrasts)

		#Decide column names based on yaxis / color_by / sort_by
		columns = [yaxis, color_by, size_by]
		for i, name in enumerate(columns): 
			if name is not None:
				if name not in self.rules.columns: #Assume that this is the contrast suffix
					columns[i] = "{0}/{1}_{2}".format(*contrast, name)
		yaxis, color_by, size_by = columns

		#Select top/bottom n rules
		
		sorted_table = self.rules.sort_values(yaxis, ascending=False)
		sorted_table.index = sorted_table["TF1"] + " + " + sorted_table["TF2"]

		top_rules = sorted_table.head(n_rules)
		bottom_rules = sorted_table.tail(n_rules)

		# Draw each cell as a scatter point with varying size and color
		fig, (ax1, ax2) = plt.subplots(2, constrained_layout=True) 

		tfcomb.plotting.bubble(data=top_rules, yaxis=yaxis, color_by=color_by, size_by=size_by, ax=ax1, **kwargs)
		tfcomb.plotting.bubble(data=bottom_rules, yaxis=yaxis, color_by=color_by, size_by=size_by, ax=ax2, **kwargs)

		#Remove x-axis label for upper plot

		#return(fig)

	def plot_network(self, contrast=None,
							color_node_by=None, 
							size_node_by=None, 
							color_edge_by="cosine_log2fc", 
							size_edge_by=None, 
							**kwargs):
		"""
		Plot the network of differential co-occurring TFs.
		
		Parameters
		-----------
		contrast : tuple
			Name of the contrast to use in tuple format e.g. (<prefix1>,<prefix2>). Default: None (the first contrast is shown).
		color_node_by : str, optional
			Name of measure to color node by. If column is not in .rules, the name will be internally converted to "prefix1/prefix2_<color_edge_by>". Default: None.
		size_node_by : str, optional
			Column in .rules to size_node_by. If column is not in .rules, the name will be internally converted to "prefix1/prefix2_<size_node_by>" Default: None. 
		color_edge_by : str, optional
			The name of measure or column to color edge by (will be internally converted to "prefix1/prefix2_<color_edge_by>"). Default: "cosine_log2fc".
		size_edge_by : str, optional
			The name of measure or column to size edge by. 
		kwargs : arguments
			Any additional arguments are passed to tfcomb.plotting.network.

		Returns
		--------
			dot network object

		See also
		---------
		tfcomb.plotting.network
		"""

		contrast = tfcomb.utils.set_contrast(contrast, self.contrasts)
		
		#Adjust column names
		self.logger.debug("Adjusting column names")
		columns = [color_node_by, size_node_by, color_edge_by, size_edge_by]
		for i, name in enumerate(columns): 
			if name is not None:
				if name not in self.rules.columns: #Assume that this is the contrast suffix
					columns[i] = "{0}/{1}_{2}".format(*contrast, name)
		color_node_by, size_node_by, color_edge_by, size_edge_by = columns
		self.logger.debug("color_node_by: {0} | size_node_by: {1} | color_edge_by: {2} | size_edge_by: {3}".format(color_node_by,
																													size_node_by,
																												   color_edge_by,
																												   size_edge_by,
																													))

		#Create subset of rules
		selected = self.rules

		#Build network
		self.logger.debug("Building network using 'tfcomb.network.build_network'")
		self.build_network() #adds .network to self
		#G = tfcomb.network.build_nx_network(selected)
		#self.network = G
		
		#Plot network
		self.logger.debug("Plotting network using 'tfcomb.plotting.network'")
		dot = tfcomb.plotting.network(self.network, color_node_by=color_node_by, size_node_by=size_node_by, 
										 color_edge_by=color_edge_by, size_edge_by=size_edge_by, 
										 verbosity=self.verbosity, **kwargs)

		return(dot)

	# -------------------------------------------------------------------------------#
	# ----------------------------- Save / import object ----------------------------#
	# -------------------------------------------------------------------------------#

	def to_pickle(self, path : str):
		""" Save the DiffCombObj to a pickle file.

		Parameters
		----------
		path : str
			Path to the output pickle file e.g. 'my_diff_comb_obj.pkl'.

		See also
		---------
		from_pickle
		"""

		f_out = open(path, 'wb')
		dill.dump(self, f_out)

	def from_pickle(self, path : str):
		"""
		Import a DiffCombObj from a pickle file.

		Parameters
		-----------
		path : str
			Path to an existing pickle file to read.

		Raises
		-------
		InputError
			If read object is not an instance of DiffCombObj.

		See also
		----------
		to_pickle
		"""

		filehandler = open(path, 'rb')
		obj = dill.load(filehandler)

		# Check if object is CombObj
		if not isinstance(obj, DiffCombObj):
			raise InputError("Object from '{0}' is not a DiffCombObj".format(path))

		# Overwrite self with DiffCombObj
		self = obj
		self.set_verbosity(self.verbosity)  # restart logger

		return self

###################################################################################
############################## Distance analysis ##################################
###################################################################################

class DistObj():
	"""
	The main class for analyzing preferred binding distances for co-occurring TFs.

	Examples
	----------

	>>> D = tfcomb.distances.DistObj()

	# Verbosity of the output log can be set using the 'verbosity' parameter:
	>>> D = tfcomb.distances.DistObj(verbosity=2)

	""" 

	#-------------------------------------------------------------------------------------------#
	#-------------------------------- Setup and sanity checks ----------------------------------#
	#-------------------------------------------------------------------------------------------#

	def __init__(self, verbosity=1): #set verbosity 

		#Function and run parameters
		self.verbosity = verbosity  #0: error, 1:info, 2:debug, 3:spam-debug
		self.logger = TFcombLogger(self.verbosity)
		
		#Variables for storing data
		self.rules = None  		     # Filled in by .fill_rules()
		self.TF_names = []		     # List of TF names
		self.TF_table = None		 # List of all TF counts
		self._raw = None             # Raw distance data [Pandas DataFrame of size n_pairs x maxDist]
		self.network = None			 # Network obj

		self.distances = None 	     # Pandas DataFrame of size n_pairs x maxDist
		self.uncorrected = None      # Pandas DataFrame of size n_pairs x maxDist
		self.corrected = None        # Pandas DataFrame of size n_pairs x maxDist
		self.linres = None           # Pandas DataFrame of size n_pairs x 3
		self.normalized = None       # Pandas DataFrame of size n_pairs x maxDist
		self.smoothed = None         # Pandas DataFrame of size n_pairs x maxDist
		self.peaks = None 	         # Pandas DataFrame of size n_pairs x n_preferredDistance 
		self.classified = None 	     # Pandas DataFrame of size n_pairs x 3 
		self.datasource = None 	     # Holds the datastructure 

		self.peaking_count = None    # Number of pairs with at least one peak 
		self.zscores = None			 # calculated zscores
		self.stringency = None       # stringency param
		self.prominence = None       # zscore or array of flat values
		self._noise_method = None 	 # private storage noise_method
		self._height_multiplier = None # private storage height_mulitplier
		self._collapsed = None 		#stores the negative values to be able to expand negative section again
		self._collapsed_peaks = None #stores the negative peaks to be able to expand negative section again
		
		self.n_bp = 0			     # Predicted number of baskets 
		self.TFBS = RegionList()     # None RegionList() of TFBS
		self.smooth_window = 1       # Smoothing window size, 1 = no smoothing

		# str <-> int encoding dicts
		self.name_to_idx = None      # Mapping TF-names: string <-> int 
		self.pair_to_idx = None      # Mapping Pairs: tuple(string) <-> int
		self.anchor_modes = {"inner": 0, "outer": 1, "center": 2} #str -> integer anchor mode

		# Default analysis parameters
		self.min_dist = 0            # Minimum distance. Default: 0
		self.max_dist = 100          # Maximum distance. Default 100.
		self.max_overlap = 0         # Maximum overlap. Default 0.
		self.directional = False     # True if direction is taken into account, false otherwise 
		self.anchor = "inner"		 # How to count distances: inner, outer or center.
		self.percentage = False		 # Whether to count distances as bp or percentages

		# private constants
		self._XLBL_ROTATION = 90    # label rotation degree for plotting x labels
		self._XLBL_FONTSIZE = 10    # label fontsize adjustment for plotting x labels

		#Use functions from CombObj
		self._set_combobj_functions()

	def _set_combobj_functions(self):
		""" Reuse CombObj functions"""

		self.copy = lambda : CombObj.copy(self)
		self.set_verbosity = lambda *args, **kwargs: CombObj.set_verbosity(self, *args, **kwargs)
		self._prepare_TFBS = lambda *args, **kwargs: CombObj._prepare_TFBS(self, *args, **kwargs)
		self._get_sort_idx = lambda *args, **kwargs: CombObj._get_sort_idx(*args, **kwargs)
		self.check_pair = lambda *args: CombObj.check_pair(self, *args)

	def __str__(self):
		""" Returns a string representation of the DistObj depending on what variables are already stored """
		
		s = "<DistObj"

		if self.TFBS is not None:
			s += ": {0} TFBS ({1} unique names)".format(len(self.TFBS), len(self.TF_names)) 

			if self.rules is not None:
				s += " | Market basket analysis: {0} rules".format(self.rules.shape[0])

				if self.peaks is not None:
					s += " | Found peaks: {0}".format(self.peaks.shape[0])
					s += " | from {0} pairs.".format(self.peaking_count)
		s += ">"
		return(s)
	
	def set_verbosity(self, level):
		""" Set the verbosity level for logging after creating the CombObj.

		Parameters
		----------
		level : int
			A value between 0-3 where 0 (only errors), 1 (info), 2 (debug), 3 (spam debug). 
		
		Returns
		-------
		None 
			Sets the verbosity level for the Logger inplace
		"""

		self.verbosity = level
		self.logger = TFcombLogger(self.verbosity) #restart logger with new verbosity	    
	
	def fill_rules(self, comb_obj):
		""" Fill DistanceObject according to reference object with all needed Values and parameters
		to perform standard prefered distance analysis

		Parameters
		----------
		comb_obj : tfcomb.objects (or any other object contain all necessary rules)
			Object from which the rules and parameters should be copied from

		Returns
		-------
		None 
			Copies values and parameters from a combObj or diffCombObj.
		
		"""

		# Check for mandatory attributes
		missing = []
		for attr in ["rules", "TF_names", "TFBS", "TF_table"]:
			try:
				getattr(comb_obj, attr)
			except AttributeError:
				missing.append(attr)
		
		if len(missing) > 0:
			raise InputError(f"Mandatory attributes {missing} missing for object {comb_obj}")
		
		#Copy internal _sites (integer representation of TFBS) and name_to_idx for name translation
		for attr in ["name_to_idx", "_sites"]:
			if hasattr(comb_obj, attr):
				setattr(self, attr, getattr(comb_obj, attr))

		# copy required attributes
		self.rules = comb_obj.rules
		self.TF_names = comb_obj.TF_names
		self.TFBS = comb_obj.TFBS
		self.TF_table = comb_obj.TF_table

		# Overwrite default parameters with values from CombObj
		variables = ["min_dist", "max_dist", "min_overlap", "max_overlap", "directional", "stranded", "anchor"]
		for variable in variables:
			if hasattr(comb_obj, variable):
				setattr(self, variable, getattr(comb_obj, variable))
	
	def reset_signal(self):
		""" Resets the signals to their original state. 

		Returns
		--------
		None 
			Resets the object datasource variable to the original raw distances
		"""

		self.logger.info("Resetting signals")
		self.datasource = self.distances

	#-------------------------------------------------------------------------------------------#
	#---------------------------------- Checks on variables-------------------------------------#
	#-------------------------------------------------------------------------------------------#

	def check_datasource(self, att):
		""" Utility function to check if distances in .<att> were set. If not, InputError is raised. 
		
		Parameters
		----------
		att : str
			Attribute name for a dataframe in self.
		"""

		df = getattr(self, att) #fetched dataframe for this att
		if df is None:

			if att == "distances" or att == "datasource":
				raise InputError("No distances evaluated yet. Please run .count_distances() first.")
			elif att == "corrected" or att == "uncorrected":
				raise InputError("Distances are not corrected yet. Please run .correct_background() first.")
			elif att == "scaled":
				raise InputError("Distance are not yet scaled. Please run .scale() first.")
			elif att == "smoothed":
				raise InputError("Distances are not yet smoothed. Please run .smooth() first.")
			elif att == "zscores":
				raise InputError("Distances were not analyzed yet. Please run .analyze_signal_all() first.")

		#If self.<att> is present, check if it is a Dataframe
		tfcomb.utils.check_type(df, pd.DataFrame, att)

	
	def check_peaks(self):
		""" Utility function to check if peaks were called. If not, InputError is raised. """

		if self.peaks is None:
			raise InputError("Peaks not evaluated yet. Please run .analyze_signal_all() first.")
			
		#If self.peaks is present, check if it is a Dataframe
		tfcomb.utils.check_type(self.peaks, pd.DataFrame, ".corrected")
	
	def check_min_max_dist(self):
		""" Utility function to check if min and max distance are valid. """
		
		if self.min_dist is None:
			raise InputError(".min_dist is not set")
		if self.max_dist is None:
			raise InputError(".max_dist is not set")
		tfcomb.utils.check_value(self.min_dist, integer=True, name=".min_dist")
		tfcomb.utils.check_value(self.max_dist, integer=True, name=".max_dist")
		
		if self.min_dist > self.max_dist:
			raise InputError(".min_dist must be lesser or equal .max_dist")


	#-------------------------------------------------------------------------------------------#
	#------------------------ Running different functions in chunks ----------------------------#
	#-------------------------------------------------------------------------------------------#

	@staticmethod
	def chunk_table(table, n):
		""" Split a pandas dataframe row-wise into n chunks.
		
		Parameters
		------------
		n : int
			A positive number of chunks to split table into.

		Returns
		--------
			list of pd.DataFrames
		"""

		n_rows = table.shape[0]
		chunk_size = math.ceil(n_rows/n) # last chunk will be chunks - (threads - 1) smaller
		chunks = [table.iloc[i:i+chunk_size,:] for i in range(0, n_rows, chunk_size)]	#list of dataframes

		return chunks

	def _multiprocess_chunks(self, threads, func, datatable):
		"""
		Split Data in chunks to multiprocess it. Because of the rather short but numerous calls mp.Pool() creates to much overhead. 
		So instead utilize chunks. 
		Following functions can be multiprocessed: 
		["analyze_signal", "evaluate_noise"]

		Parameters
		----------
		threads : int
			Number of threads used
		func : tfcomb.utils.*_chunks function
			Related chunk process function from tfcomb.utils
		datatable: pd.DataFrame
			Datatable to operate on (e.g. .distances)

		See also
		-------
		tfcomb.utils.analyze_signal_chunks
		tfcomb.utils.evaluate_noise_chunks

		"""
		
		# check function is supported
		if not callable(func):
			raise InputError(f"Input {func} not callable. Please provide a function")
		check_string(func.__name__,["analyze_signal_chunks", 
									"evaluate_noise_chunks"], name="function")

		self.logger.debug(f"Multiprocessing chunks for {func}")
		
		# open Pool for multiprocessing
		pool = mp.Pool(threads)

		#Chunk table into threads. multiprocess every function call will result in mp overhead, therefore chunk it
		chunks = self.chunk_table(datatable, threads)
		
		# start one chunk per thread
		jobs = []
		for chunk in chunks: #range(threads):

			#Apply function with parameters and add to pool
			if func == tfcomb.utils.analyze_signal_chunks:
				job = pool.apply_async(func, args=(chunk, self.threshold, )) # apply function with params

			elif func == tfcomb.utils.evaluate_noise_chunks:
				# subset peaks to those in chunk
				peaks_sub = self.peaks.loc[chunk.index.tolist()]
				job = pool.apply_async(func, args=(chunk, peaks_sub, self._noise_method, self._height_multiplier, )) # apply function with params

			jobs.append(job)

		# accept no new jobs
		pool.close()
			
		# log_progress(jobs, self.logger) # doesn't work with chunks (TODO)
		
		# get results from jobs
		results = []
		for job in jobs:
			results += job.get()
		# wait for all jobs to be finished and tidy up pools
		pool.join()

		# convert to numpy 
		results = np.array(results, dtype=object) # prevent convert float to str 
		return results
			


	#-------------------------------------------------------------------------------------------#
	#---------------------------- Counting distances between TFs -------------------------------#
	#-------------------------------------------------------------------------------------------#

	def count_distances(self, directional=None, stranded=None, percentage=False, percentage_bins=100):
		""" Count distances for co_occurring TFs, can be followed by analyze_distances
			to determine preferred binding distances

		Parameters
		----------
		directional : bool or None, optional
			Decide if direction of found pairs should be taken into account, e.g. whether  "<---TF1---> <---TF2--->" is only counted as 
			TF1-TF2 (directional=True) or also as TF2-TF1 (directional=False). If directional is None, self.directional will be used.
			Default: None.
		stranded : bool or None, optional
			Whether to take strand of TFBS into account when counting distances. If stranded is None, self.stranded will be used. 
			Default: None
		percentage : bool, optional
			Whether to count distances as bp or percentage of longest TF1/TF2 region. If True, output will be collected in 1-percent increments from 0-1. 
			If False, output depends on the min/max distance values given in the DistObj. Default: False.
		
		Returns
		--------
		None 
			Fills the object variable .distances.

		"""

		#todo; reset any previous counts

		#Replace directional/stranded with internal values
		directional = self.directional if directional is None else directional
		stranded = self.stranded if stranded is None else stranded

		#Check input types
		tfcomb.utils.check_string(self.anchor, list(self.anchor_modes.keys()), "self.anchor")
		tfcomb.utils.check_type(directional, [bool], "directional")
		tfcomb.utils.check_type(stranded, bool, "stranded")
		tfcomb.utils.check_type(percentage, bool, "percentage")
		tfcomb.utils.check_value(percentage_bins, vmin=1, integer=True, name="percentage_bins")
		self.check_min_max_dist()
		
		#Check if overlapping is allowed (when anchor == 0 (inner))):
		if self.anchor == "inner" and self.min_dist < 0 and self.max_overlap == 0:
			self.logger.warning("'min_dist' is below 0, but max_overlap is set to 0. Please set max_overlap > 0 in order to count overlapping pairs with negative distances.")

		self.logger.info("Preparing to count distances.")

		# encode chromosome,pairs and name to int representation
		self._prepare_TFBS() #prepare _sites and name_to_idx dict
		sites = self._sites
		
		#Check if rules look stranded
		rule_names = list(set(self.rules["TF1"].tolist() + self.rules["TF2"].tolist()))
		strand_matching = [re.match("(.+)\(([+-.])\)$", name) for name in rule_names] #check if name is "NAME(+/./-)"
		n_strand_names = sum([match is not None for match in strand_matching])
		if n_strand_names > 0 and stranded == False:
			raise InputError("TF names in .rules contain strand information, but stranded is set to False. Please set stranded to True to count distances for rules.")

		#Should strand be taken into account?
		if stranded == True:
			sites = self._sites.copy() #don't change self._sites

			name_to_idx = {}
			TF_names = [] #names in order of idx
			current_idx = -1

			for i, site in enumerate(self.TFBS):
				name = "{0}({1})".format(site.name, site.strand)
				if name not in name_to_idx:
					TF_names.append(name)
					current_idx = current_idx + 1
					name_to_idx[name] = current_idx
				sites[i][-1] = name_to_idx[name] #set new idx based on stranded name
			
		else:
			TF_names = self.TF_names
			name_to_idx = self.name_to_idx

		self.pairs = [(name_to_idx[TF1], name_to_idx[TF2]) for (TF1, TF2) in self.rules[["TF1", "TF2"]].values]	#list of TF1/TF rules to count distances for
		self.logger.spam("TF_names: {0}".format(TF_names))

		#Sort sites by mid if anchor == center:
		if self.anchor == "center": 
			sort_idx = self._get_sort_idx(sites, anchor="center")
			sites = sites[sort_idx, :]

		self.logger.info("Calculating distances")
		anchor_mode = self.anchor_modes[self.anchor]
		self._raw = count_co_occurrence(sites, 
											min_dist=self.min_dist,
											max_dist=self.max_dist,
											min_overlap=self.min_overlap,
											max_overlap=self.max_overlap,
											binarize=False,				#does not affect distance counting
											anchor=anchor_mode,			#integer representation of anchor
											n_names=len(TF_names),
											directional=directional,
											task=2,						#task = count distances
											rules=self.pairs,			#rules to count distances for
											percentage=percentage,
											percentage_bins=percentage_bins,
											)
		self.percentage = percentage
		self.percentage_bins = percentage_bins

		# convert raw counts (numpy array with int encoded pair names) to better readable format (pandas DataFrame with TF names)
		self._raw_to_human_readable(name_to_idx) #fills in .distances

		self.logger.info("Done finding distances! Results are found in .distances")
		self.logger.info("You can now run .smooth() and/or .correct_background() to preprocess sites before finding peaks.")
		self.logger.info("Or you can find peaks directly using .analyze_signal_all()")
	
	def _raw_to_human_readable(self, name_to_idx):
		""" Get the raw distance in human readable format. Sets the variable '.distances' which is a pd.Dataframe with the columns:
			TF1 name, TF2 name, count min_dist, count min_dist +1, ...., count max_dist).

			Note: 
			Normalization method is min_max normalization: (x[i] - min(x))/(max(x)-min(x))

			Parameters
			-----------
			name_to_idx : dict
				Dictionary encoding TF names to integers
		"""

		#Converting to pandas format 
		self.logger.debug("Converting raw count data to pretty dataframe")

		if self.percentage == False:
			columns = ['TF1', 'TF2'] + list(range(self.min_dist, self.max_dist + 1))
		else:
			columns = ['TF1', 'TF2'] + list(range(self.percentage_bins+1))

		self.distances = pd.DataFrame(self._raw, columns=columns)
		
		#Convert integers to TF names
		# get names from int encoding
		idx_to_name = {}
		for name, idx in name_to_idx.items():
			idx_to_name[idx] = name 

		self.distances["TF1"] = [idx_to_name[idx] for idx in self.distances["TF1"]]
		self.distances["TF2"] = [idx_to_name[idx] for idx in self.distances["TF2"]]
		self.distances.index = self.distances["TF1"] + "-" + self.distances["TF2"]

		#Set datasource for future normalization/correction/analysis
		self.datasource = self.distances.copy()


	#-------------------------------------------------------------------------------------------#
	#------------------- Process counted distances (scale/smooth/correct)-----------------------#
	#-------------------------------------------------------------------------------------------#
	
	#Scale signal
	def scale(self, how="min-max"):
		""" Scale the counted distances per pair. Saves the scaled counts into .scaled and updates .datasource. 
		
		Parameters
		-----------
		how : str, optional
			How to scale the counts. Must be one of: ["min-max", "fraction"]. If "min-max", all counts are scaled between 0 and 1. 
			If "fraction", the sum of all counts are scled between 0 and 1. Default: "min-max".
		"""

		check_string(how, ["min-max", "fraction"], "how")

		distances_mat = self.datasource.iloc[:,2:].to_numpy().astype(float) #float for nan replacement

		#Calculate scaling depending on "how"
		if how == "min-max":

			min_count = np.array([distances_mat.min(axis=1)]).T #convert to column vectors
			max_count = np.array([distances_mat.max(axis=1)]).T #convert to column vectors
			ranges = max_count - min_count #min-max range per pair
			ranges[ranges==0] = np.nan

			#Perform scaling and save to self.scaled
			scaled_mat = (distances_mat - min_count) / ranges
			scaled_mat = np.nan_to_num(scaled_mat)

		elif how == "fraction":
			
			sum_count = distances_mat.sum(axis=1) #sum of each row
			sum_count[sum_count==0] = np.nan

			scaled_mat = distances_mat / sum_count.reshape(-1,1)
			scaled_mat = np.nan_to_num(scaled_mat) #rescale back to 0

		#Set .scaled variable
		self.scaled = self.datasource.copy()
		self.scaled.iloc[:,2:] = scaled_mat

		#Update datasource
		self.datasource = self.scaled

	#Smooth signals
	def smooth(self, window_size=3, reduce=True):
		""" Helper function for smoothing all rules with a given window size.
			
		Parameters
		----------
		window_size : int, optional 
			Window size for the rolling smoothing window. A bigger window produces larger flanking ranks at the sides. Default: 3.
		reduce : bool, optional
			Reduce the distances to the positions with a full window, i.e. if the window size is 3, the first and last distances are removed.
			This prevents flawed enrichment of peaks at the borders of the distances. Default: True. 

		Returns
		--------
		None 
			Fills the object variable .smoothed and updates .datasource
		"""
		
		tfcomb.utils.check_value(window_size, vmin=0, integer=True, name="window size")
	
		if self.is_smoothed() == True:
			self.logger.warning("Data was already smoothed - beware that smoothing again might produce unwanted results. Please use .reset_signal() to reset the signal.")

		self.logger.info(f"Smoothing signals with window size {window_size}")
		mat = self.datasource.iloc[:,2:].to_numpy() #distances counted
		distances = self.datasource.columns[2:]
		smoothed_mat = np.array([tfcomb.counting.rolling_mean(mat[row,:].astype(float), window_size) for row in range(len(mat))]) #smooth values per pair
		
		#Cut the borders of smoothed mat if reduce == True
		if reduce == True:
			lf = int(np.floor((window_size - 1) / 2.0))
			rf = int(np.ceil((window_size - 1)/ 2.0))
			smoothed_mat = smoothed_mat[:,lf:-rf]
			distances = distances[lf:-rf]
		
		#Create final table
		smoothed_table = pd.DataFrame(smoothed_mat, columns=distances).reset_index(drop=True)
		self.smoothed = pd.concat([self.datasource.iloc[:,:2].reset_index(drop=True), smoothed_table], axis=1)
		self.smoothed.index = self.datasource.index #reset index

		#Save information about smoothing to object
		self.smooth_window = window_size
		self.datasource = self.smoothed

	def is_smoothed(self):
		""" Return True if data was smoothed during analysis, False otherwise
			
		Returns
		--------
		bool 
			True if smoothed, False otherwiese
		"""
		
		if (self.smoothed is None) or (self.smooth_window <= 1): 
			return False
		return True

	#Correct background signal from distance counts
	def correct_background(self, frac=0.66, threads=1):
		""" Corrects the background of distances.
		
		Parameters
		-------------
		frac : float, optional
			Fraction of data used to calculate smooth. Setting this fraction lower will cause stronger smoothing effect. Default: 0.66 
		threads : int, optional
			Number of threads to use in functions. Default: 1.

		Returns
		----------
		None 
			Fills the object variable .corrected
		"""

		#Check input parameters
		self.check_datasource("distances")
		tfcomb.utils.check_value(frac, vmin=0, vmax=1, name="frac")
		tfcomb.utils.check_value(threads, vmin=1, vmax=os.cpu_count(), integer=True, name="threads")

		#Correct background in chunks
		pool = mp.Pool(threads)
		chunks = self.chunk_table(self.datasource, threads)
		jobs = []
		for chunk in chunks:
			job = pool.apply_async(self._correct_chunk, (chunk, frac, ))
			jobs.append(job)

		#Collect results
		pool.close()
		pool.join() #waits for all jobs
		results = [job.get() for job in jobs]

		#Join tables
		self.uncorrected = self.datasource #used for plotting
		self.lowess = pd.concat([result[0] for result in results])
		self.corrected = pd.concat([result[1] for result in results])
		self.datasource = self.corrected #update datasource

		self.logger.info("Background correction finished! Results can be found in .corrected")

	@staticmethod
	def _correct_chunk(counts, frac=0.66):
		""" Correct counts for background signal using lowess smoothing.
		
		Parameters
		------------
		counts : pd.DataFrame
			Subset of .datasource dataframe
		frac : float, optional
			Fraction of data used to calculate smooth. Default: 0.66.

		Returns
		---------
		Tuple of two dataframes
			counts_lowess : lowess smoothing of counts
			corrected : counts corrected
		"""

		#Run lowess smoothing for each row in input counts
		n_rows = counts.shape[0]
		counts_lowess = counts.copy()
		for i in range(n_rows):
			signal = counts.iloc[i,2:] #first two columns are TF names
			counts_lowess.iloc[i,2:] = sm.nonparametric.lowess(signal, range(len(signal)), frac=frac)[:,1] #first column in result is x-values

		#Correct counts
		corrected = counts.copy()
		corrected.iloc[:,2:] = counts.iloc[:,2:] - counts_lowess.iloc[:,2:] #correct by subtracting lowess

		return (counts_lowess, corrected)

	#-------------------------------------------------------------------------------------------#
	#--------------------------- Analysis to get preferred distances ---------------------------#
	#-------------------------------------------------------------------------------------------#

	def _get_zscores(self, datasource, mixture=False, threads=1):
		"""
		Calculate zscores for the data in datasource. 

		Parameters
		--------------
		datasource : pd.DataFrame
			The datasource table to calculate zscore on (can be a subset of all rules).
		mixture : bool, optional
			Whether to estimate zscore using all datapoints (False) or to use a mixture model (True). Default: False. 
		threads : int, optional
			Find z-scores using multiprocessing. 
		"""
		#Get zscores in chunks
		pool = mp.Pool(threads)
		chunks = self.chunk_table(datasource, threads)
		jobs = []
		for chunk in chunks:
			job = pool.apply_async(self._zscore_chunk, (chunk, mixture, ))
			jobs.append(job)

		#Collect results
		pool.close()
		pool.join() #waits for all jobs
		results = [job.get() for job in jobs]

		#Each result consists of the zscore table and a dict containing backgruond information
		self.zscores = pd.concat([result[0] for result in results])
		self.bg_dist = {}
		for d in [result[1] for result in results]:
			self.bg_dist.update(d) #add results to bg_dist

	@staticmethod
	def _zscore_chunk(datasource, mixture=False):

		#Fetch information from data
		indices = datasource.index.tolist() #all index names e.g. TF1-TF2
		distance_cols = datasource.columns.tolist()[2:]

		#Regular zscore
		stds = datasource[distance_cols].std(ddof=0, axis=1)
		stds[stds == 0] = np.nan #possible divide by zero in zscore calculation
		means = datasource[distance_cols].mean(axis=1)

		#save information on background distribution
		bg_dist = {}
		for i, idx in enumerate(indices):
			bg_dist[idx] = [(means[i], stds[i], 1.0)] #mean, std, weight for each pair

		# Calculate zscore (x-mean)/std
		zsc = datasource[distance_cols].subtract(means, axis=0).divide(stds, axis=0).fillna(0) 

		#Save zscores to object
		zscores = datasource.copy()
		zscores.iloc[:,2:] = zsc

		#If mixture == True, try to update with mixture model zscores
		if mixture == True:
		
			for idx in datasource.index.tolist(): #index name e.g. TF1-TF2

				X = datasource.loc[idx].iloc[2:].values.astype(float)

				#Fit kernel to input data to increase signal
				kernel = stats.gaussian_kde(X)
				vals = kernel.resample(1000).reshape(-1, 1)

				#The data is either 1 or two components; find which one
				models = []
				for n in [1,2]:
					try:
						model = GaussianMixture(n).fit(vals)
						models.append(model)
					except:
						pass
				
				#If it was possible to fit one/two components; else, zscore stays original
				if len(models) > 0:
					AIC = [m.aic(X.reshape(-1, 1)) for m in models]

					M_best = models[np.argmin(AIC)] #model with lowest AIC (1 or two components)
					means = M_best.means_.ravel()
					stds =  np.sqrt(M_best.covariances_.ravel())
					weights = M_best.weights_

					#Sort components based no weight (largest component is background)
					idx_sort = np.argsort(-M_best.weights_)

					#Add components to bg_dist in order
					bg_dist[idx] = []
					for i in idx_sort:
						bg_dist[idx].append((means[i], stds[i], weights[i]))

					dist_mean, dist_std, _ = bg_dist[idx][0] #the first distribution (largest component) is used as background
					zscore = (X - dist_mean) / dist_std
					zscores.loc[idx, distance_cols] = zscore #update ztscores for this idx

		return (zscores, bg_dist)

	def analyze_signal_all(self, threads=1, method="zscore", threshold=2, min_count=1, save=None):
		""" 
		After background correction is done, the signal is analyzed for peaks, 
		indicating preferred binding distances. There can be more than one peak (more than one preferred binding distance) per 
		Signal. Peaks are called with scipy.signal.find_peaks().
		
		Parameters
		----------
		threads : int
			Number of threads used. Default: 1.
		method : str
			Method for transforming counts. Can be one of: "zscore" or "flat". 
			If "zscore", the zscore for the pairs is used.
			If "flat", no transformation is performed.
			Default: "zscore".
		threshold : float
			The lower threshold for selecting peaks. Default: 2.
		min_count : int
			Minimum count of TF1-TF2 occurrences for a preferred distance to be called. Default: 1 (all occurrences are considered).
		save : str
			Path to save the peaks table to. Default: None (table is not written).

		Returns
		-------
		None 
			Fills the object variable self.peaks, self.peaking_count
		"""

		# Check given input	
		if save is not None:
			tfcomb.utils.check_writeability(save)
		tfcomb.utils.check_value(threshold, name="threshold")
		tfcomb.utils.check_string(method, ["zscore", "flat"], name="method")

		# sanity checks
		self.check_datasource("distances")
		

		#----- Find preferred peaks -----#
		self.logger.info(f"Analyzing Signal with threads {threads}")
	
		datasource = self.datasource
		distance_cols = datasource.columns[2:].tolist()

		#Set threshold on the number of sites
		if min_count > 1:
			n = datasource.shape[0]
			counts = self.distances[distance_cols].sum(axis=1) #raw distances for counting sites
			datasource = datasource.loc[counts >= min_count, :]
			self.logger.info("Reduced number of rules from {0} to {1} having min_count >= {2}".format(n, datasource.shape[0], min_count))

		# save params
		self.method = method
		self.threshold = threshold

		#Set input to functions 
		self.logger.info("Calculating zscores for signals")
		if method == "zscore":
			self._get_zscores(datasource, threads=threads)
			signal = self.zscores

		elif method == "zscore-mixture": #under development; not applicable at the moment
			#If "zscore-mixture", a modified z-score on a Gaussian Mixture Model of counts is used.
			self._get_zscores(datasource, mixture=True, threads=threads)
			signal = self.zscores

		elif method == "flat":
			signal = datasource #no change in signal

		#Find peaks from input signal
		self.logger.info("Finding preferred distances")
		res = self._multiprocess_chunks(threads, tfcomb.utils.analyze_signal_chunks, signal)

		#Collect results from all pairs to pandas dataframe
		res = pd.concat([pd.DataFrame(pair_res) for pair_res in res]).reset_index(drop=True)

		#Format order of columns
		columns = ["TF1", "TF2", "Distance", "peak_heights", "prominences", "Threshold"]
		res = res[columns]
		res["Distance"] = res["Distance"].astype(int) # distances are float - change to int
		res.rename(columns={"peak_heights":"Peak Heights",
							"prominences": "Prominences"}, inplace=True)
		self.peaks = res

		# Add total counts of (TF1-TF2) to each peak
		self.peaks.index = self.peaks["TF1"] + "-" + self.peaks["TF2"]
		counts = self.distances[distance_cols].sum(axis=1)
		counts.name = "TF1_TF2_count"
		self.peaks = self.peaks.merge(counts.to_frame(), left_index=True, right_index=True)

		#Define list of distances included in each peak (depending on smooth window)
		distances = datasource.columns[2:].tolist()
		dist_min = min(distances)
		dist_max = max(distances)
		lf = int(np.floor((self.smooth_window-1) / 2.0))
		rf = int(np.ceil((self.smooth_window-1) / 2.0)) 
		self.peaks["dist_list"] = [list(range(max(dist-lf, dist_min), min(dist+rf, dist_max)+1)) for dist in self.peaks["Distance"]]

		#Get count per peak depending on smooth window
		self.peaks["Distance_count"] = [self.distances.loc[idx, dist_list].sum() for idx, dist_list in zip(self.peaks.index, self.peaks["dist_list"])]
		self.peaks["Distance_percent"] = (self.peaks["Distance_count"] / self.peaks["TF1_TF2_count"]) * 100
	
		#Replace Distance + / - lf/rf
		self.peaks["Distance_window"] = ["[{0};{1}]".format(min(dist), max(dist)) for dist in self.peaks["dist_list"]]
		
		#Sort peaks on highest prominences
		self.peaks = self.peaks.sort_values("Prominences", ascending=False)
		self.peaks.drop(columns=["dist_list"], inplace=True) #Remove temporary column

		#----- Save stats on run -----#
		self.peaking_count = self.peaks.drop_duplicates(["TF1", "TF2"]).shape[0] #number of pairs with any peaks

		# QoL save of threshold and method
		self.thresh = self.rules[["TF1", "TF2"]] #save threshold for all rules, even those without peaks
		self.thresh["Threshold"] = threshold
		self.thresh["Method"] = method

		# Save dataframe
		if save is not None:
			self.peaks.to_csv(save)

		self.logger.info("Done analyzing signal. Results are found in .peaks")	

	#-------------------------------------------------------------------------------------------#
	#------------------------- Evaluation and ranking of found distances -----------------------#
	#-------------------------------------------------------------------------------------------#

	def evaluate_noise(self, threads=1, method="median", height_multiplier=0.75):
		"""
		Evaluates the noisiness of the signal. Therefore the peaks are cut out and the remaining signal is analyzed.

		Parameters
		---------
		threads : int
			Number of threads used for evaluation.
			Default: 1
		method : str
			Measurement to calculate the noisiness of a signal.
			One of ["median", "min_max"].
			Default: "median"
		height_multiplier : float
			Height multiplier (percentage) to calculate cut points. Must be between 0 and 1.
			Default: 0.75

		See also 
		--------
		tfcomb.utils.evaluate_noise_chunks
		"""
		self.check_peaks()

		self._noise_method = method
		self._height_multiplier = height_multiplier
		
		datasource = self.datasource

		#Subset datasource to TFs with peaks
		peak_pairs = set(self.peaks.index)
		datasource = datasource[datasource.index.isin(peak_pairs)]

		#Evaluate noisiness
		self.logger.info(f"Evaluating noisiness of the signals with {threads} threads")
		res = self._multiprocess_chunks(threads, tfcomb.utils.evaluate_noise_chunks, datasource)
		noisiness = pd.DataFrame(res, columns=["TF1", "TF2", "Noisiness"])
		
		# merge noisiness
		self.peaks = self.peaks.merge(noisiness)
		self.peaks.index = self.peaks["TF1"] + "-" + self.peaks["TF2"]

	
	def rank_rules(self, by=["Distance_percent", "Peak Heights", "Noisiness"], calc_mean=True):
		""" 
		ranks rules within each column specified. 

		Parameters
		----------
		by : list of strings
			Columns for wich the rules should be ranked
			Default: ["Distance_percent", "Peak Heights", "Noisiness"]
		calc_mean : bool
			True if an extra column should be calculated containing the mean rank, false otherwise
			Default: True
			
		Raises
		-------
		InputError
			If columns selection (parameter: by) is not valid.

		Returns
		-------
		None 
			adds a rank column for each criteria given plus one for the mean if set to True
		"""

		# check peaks are calculated + by are only valid columns
		self.check_peaks()
		if (not set(by).issubset(self.peaks.columns)):
			raise InputError(f"Column selection not valid. Possible column names to rank by: {self.peaks.columns.values}")

		# save col names for mean_rank
		rank_cols = []
		# rank all given columns
		for col in by:
			# new column name
			rank_col = "rank_" + col
			# decide if biggest number = rank 1 or rank n
			if col =="Noisiness":
				self.peaks[rank_col] = self.peaks[col].rank(method="dense", ascending=1) 
			else:
				self.peaks[rank_col] = self.peaks[col].rank(method="dense", ascending=0)
			rank_cols.append(rank_col)

		#Add mean rank to peaks
		if calc_mean:
			# calculate mean rank (from all column ranks)
			self.peaks["mean_rank"] = self.peaks[rank_cols].mean(axis=1)
			# nice to have the best ranks at the top 
			self.peaks = self.peaks.sort_values(by="mean_rank")
		
	#-------------------------------------------------------------------------------------------#
	#------------------- Additional functionality/analysis for distances -----------------------#
	#-------------------------------------------------------------------------------------------#

	def mean_distance(self, source="datasource"):
		""" Get the mean distance for each rule in .rules.
		
		Returns
		---------
		pandas.DataFrame containing "mean_distance" per rule.

		""" 

		self.check_datasource(source)
		datasource = getattr(self, source)

		#Calculate weighted average
		sums = datasource.iloc[:,2:].sum(axis=1).values
		weights = datasource.iloc[:,2:].div(sums, axis=0)
		distances = datasource.columns[2:].tolist()
		avg = (weights * distances).sum(axis=1)

		#Convert series to df
		df = pd.DataFrame(avg).rename(columns={0:"mean_distance"})

		return df

	def max_distance(self, source="datasource"):
		""" Get the distance with the maximum signal for each rule in .rules.
		
		Parameters
		-----------
		source : str
			The name of the datasource to use for calculation. Default: "datasource" (the current state of data).

		Returns
		---------
		pandas.DataFrame containing "max_distance" per rule.
		"""

		self.check_datasource(source)
		datasource = getattr(self, source)

		df = datasource.iloc[:,2:].idxmax(axis=1).to_frame() #idxmax directly gives the name of column
		df.columns = ["max_distance"]
		
		return df


	def analyze_hubs(self):
		""" 
		Counts the number of different partners each transcription factor forms a peak with, **with at least one peak**.

		Returns
		--------
		pd.Series 
			A panda series with the tf as index and the count as integer
		"""
		
		self.check_peaks()

		occurrences= collections.Counter([x for (x,z) in set(self.peaks.set_index(["TF1","TF2"]).index)])
		
		return pd.Series(occurrences)		

	def count_peaks(self):
		"""
		Counts the number of identified distance peaks per rules. 

		Returns
		---------
		pd.DataFrame
			A dataframe containing 'n_peaks' (column) for each TF1-TF2 rule (index)
		"""
		self.check_peaks()

		peak_counts = self.peaks.groupby(["TF1", "TF2"]).size().to_frame().rename(columns={0:"n_peaks"}).reset_index()
		peak_counts.index = peak_counts["TF1"] + "-" + peak_counts["TF2"]
		peak_counts.drop(columns=["TF1", "TF2"], inplace=True)

		#Merge peak counts with rules
		counts = self.rules[["TF1", "TF2"]]
		counts = counts.merge(peak_counts, left_index=True, right_index=True, how="left")
		counts = counts.fillna(0) #Fill with 0 for rules without peaks
		counts["n_peaks"] = counts["n_peaks"].astype(int)

		return(counts)

	def classify_rules(self):
		""" 
		Classify all rules True if at least one peak was found, False otherwise.  

		Returns
		--------
		None 
			fills .classified
		"""

		self.check_peaks()

		self.logger.info("classifying rules")

		p_index = self.peaks.set_index(["TF1","TF2"]).index.drop_duplicates()
		
		datasource = self.datasource[["TF1","TF2"]]

		datasource["isPeaking"] = datasource.set_index(["TF1","TF2"]).index.isin(p_index)

		datasource.index = datasource["TF1"] + "-" + datasource["TF2"]

		self.classified = datasource

		self.logger.info(f"classifcation done. Results can be found in .classified")

	def get_periodicity(self):
		"""
		Calculate periodicity for all rules via autocorrelation.
		
		Returns
		---------
		None
			Fills the object variable .autocorrelation and .periodicity
		
		"""
		
		#Collect data
		datasource = self.datasource #raw/corrected/smoothed
		distances = datasource.columns.tolist()[2:]
		lags = range(1,len(distances)) #lag 1 to end of distances
		
		#Setup output table
		self.autocorrelation = pd.DataFrame(index=datasource.index, columns=lags)
		
		#Calculate autocorrelation per pair
		n_rules = datasource.shape[0]
		for i in range(n_rules): #for each pair 
			
			counts = datasource.iloc[i,2:].values

			#Calculate autocorrelation
			autocorr = sm.tsa.acf(counts, nlags=len(lags), fft=True)
			autocorr = autocorr[1:] #exclude lag 0           
			
			self.autocorrelation.iloc[i,:] = autocorr
		
		#Find best periodicity per pair
		info = {}
		for i in range(n_rules):
			
			signal = self.autocorrelation.iloc[i,:]
			
			y = np.fft.rfft(signal)
			x = np.fft.rfftfreq(len(signal))
		
			#Position of highest peak
			idx_max = np.argmax(y)

			info[self.autocorrelation.index[i]] = {"period": np.round(1/x[idx_max], 2),
												"amplitude": np.round(np.abs(y[idx_max]), 3)}
			
		self.periodicity = pd.DataFrame().from_dict(info, orient="index")

	def plot_autocorrelation(self, pair):
		"""
		Plot the autocorrelation for a pair, which shows the lag of periodicity in the counted distances.
		
		Parameters
		------------
		pair : tuple(str, str)
			TF names to plot. e.g. ("NFYA","NFYB")
		"""
		
		from statsmodels.graphics import tsaplots

		#check pair
		name = "-".join(pair)
		
		#Fetch signal for pair
		datasource = self.datasource
		signal = datasource.loc[name][2:] #raw/corrected/smoothed
		distances = datasource.columns.tolist()[2:]
		lags = range(1,len(distances)) #lag 1 to end of distances
		
		fig, ax = plt.subplots()
		fig = tsaplots.plot_acf(signal, lags=len(lags), zero=False, alpha=None, title=f"Autocorrelation for {name}", ax=ax)

		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)

		ax.set_xlabel("Lag (bp)")
		ax.set_ylabel("Correlation coefficient")

		return ax

	def build_network(self):
		""" 
		Builds a TF-TF co-occurrence network for the rules within object.
			 
		Returns
		-------
		None 
			fills the .network attribute of the `CombObj` with a networkx.Graph object

		See also
		-------
		tfcomb.network.build_nx_network
		"""

		#Build network
		self.logger.debug("Building network using tfcomb.network.build_network")
		self.network = tfcomb.network.build_network(self.peaks, node_table=self.TF_table, verbosity=self.verbosity)
		self.logger.info("Finished! The network is found within <CombObj>.<distObj>.network.")
	

	#-------------------------------------------------------------------------------------------------------------#
	#---------------------------------------------- Plotting -----------------------------------------------------#
	#-------------------------------------------------------------------------------------------------------------#
	
	def plot_bg_estimation(self, pair):
		""" Plot the background estimation for pair for debugging background estimation

		Parameters
		------------
		pair : tuple(str, str)
			TF names to plot. e.g. ("NFYA","NFYB")
		"""

		TF1, TF2 = pair
		ind = TF1 + "-" + TF2 # construct index

		#Get counts for distances
		distances = self.datasource.columns.tolist()[2:]
		counts = self.datasource.loc[ind, distances]

		#Initialize plot
		_, ax = plt.subplots(ncols=2, figsize=(10,4))

		#Plot distribution of distances
		sns.histplot(x=distances, weights=counts, ax=ax[0], bins=len(distances))
		ax[0].set_xlabel("Distance in bp")
		ax[0].set_ylabel("Counts per bp")

		#Plot hist of distances
		sns.histplot(x=counts, ax=ax[1], stat="probability", color="grey")
		ax[1].set_xlabel("Counts per distance")

		#Add background distributions
		dist_list = self.bg_dist.get(ind, [])
		for i, (mu, std, weight) in enumerate(dist_list):

			name = "Background distribution" if i == 0 else "Upper distribution"

			#Create normal dist with mu/std
			x = np.linspace(min(counts), max(counts), 100)
			rv = stats.norm(mu, std)
			pdf = rv.pdf(x)

			ax[1].plot(x, pdf, label="{0} ({1:.1f}%)".format(name, weight*100))

		#Final adjustments
		ax[1].legend()
		plt.subplots_adjust(wspace=0.3)


	def _plot_all(self, save_path, method):
		"""
		Plots all pairs.
		
		Parameters
		----------
		save_path : str
			Directory path to save plots to. Filename will be created with "{method}_{tf1}_{tf2}.png".
			e.g. save_path/linres_NFYA_NFYB.png
		method : str
			Plotting style	

		See also
		----------
		tfcomb.objects.DistObj.plot

		"""

		self.check_min_max_dist()
		self.check_datasource("distances")
		tfcomb.utils.check_dir(save_path)

		# warn user
		self.logger.info(f"Plotting {method}-plots for all pairs. This may take a while.")

		for tf1,tf2 in list(zip(self.distances.TF1, self.distances.TF2)):
			self.plot((tf1, tf2), method=method, save=os.path.join(save_path,f"{method}_{tf1}_{tf2}.png"))

	def _collapse_negative(self, sourcetable, method="max"):
		"""
		Method to collapse negative coulmns(-min distance to 0). e.g. columns [-3, -2, -1] will ne summarized as "neg".
		
		Parameters
		-----------
		sourcetable: pd.DataFrame
			DataFrame which should be collapsed (e.g. .smoothed or distances)
		method : str, optional
			Summarization Method. One of ["max","min","mean","sum"].
			e.g. with "max" 2.3 is chosen from [1.2, 2.3, 1.5], for sum: "neg" would be 5 
			Default: max
		
		Returns
		-----------
		pd.DataFrame
			collapsed dataframe for plotting purposes
		pd.DataFrame, None
			altered peak list (negative distances are replaced with "neg"). If no peaks found yet, None will be returned

		Notes
		--------
		This method is intended to alter the plots produced, not to run analysis on it. Although it is possible 
		with most of the functions, we recommend not to use collapsed data for analysis steps.  
		To revert collapsing, use .expand_negative().

		See also
		--------
		tfcomb.objects.expand_negative
		"""

		# check method
		tfcomb.utils.check_string(method, ["max","min","mean","sum"])
		# check if negative positions possible
		if self.min_dist >= 0:
			raise InputError("Data must contain negative position to collapse")

		datasource = sourcetable
		# create negative columns 
		neg_cols = neg_cols = [ x for x in range(self.min_dist,0)]
		
		#select negative part
		neg = datasource[["TF1","TF2"] + neg_cols]

		#add negative column according to method
		if method =="max":	
			neg["neg"] = neg[neg_cols].max(axis=1)
		elif method=="min":
			neg["neg"] = neg[neg_cols].min(axis=1)
		elif method=="mean":
			neg["neg"] = neg[neg_cols].mean(axis=1)
		elif method=="sum":
			neg["neg"] = neg[neg_cols].sum(axis=1)

		# remove negative columns from df
		datasource = datasource.drop(neg_cols, axis=1)
		try:
			# try inserting collapsed negative value
			datasource.insert(2, "neg", neg["neg"])
		except ValueError:
			raise InputError("Datasource already contains a negative column")

		# alter peaks list
		peaks = None
		if self.peaks is not None:
			peaks = copy.deepcopy(self.peaks)
			# replace every negative position with "neg"
			[peaks["Distance"].replace(x,"neg", inplace=True) for x in range(self.min_dist,0)]

		return (datasource, peaks)

	def plot(self, pair, method="peaks", 
						style="hist", 
						show_peaks=True, 
						save=None, 
						config=None, 
						collapse=None, 
						ax=None, 
						color='tab:blue', 
						max_dist=None,
						**kwargs):
		"""
		Produces different plots.
		
		Parameters
		-----------
		pair : tuple(str, str)
			TF names to plot. e.g. ("NFYA","NFYB")
		method : str, optional
			Plotting method. One of:
			- 'peaks': Shows the z-score signal and any peaks found by analyze_signal_all.
			- 'correction': Shows the fit of the lowess curve to the data.
			- 'datasource', 'distances', 'scaled', 'corrected', 'smoothed': Shows the signal of the counts given in the .<method> table.
			Default: 'peaks'.
		style : str, optional
			What style to plot the datasource in. Can be one of: ["hist", "kde", "line"]. Default: "hist".
		show_peaks : bool, optional
			Whether to show the identified peak(s) (if any were found) in the plot. Default: True.
		save: str, optional
			Path to save the plots to. If save is None plots won't be plotted. 
			Default: None
		config : dict, optional
			Config for some plotting methods. \n
			e.g. {"nbins":100} for histogram like plots or {"bwadjust":0.1} for kde (densitiy) plot. \n
			If set to *None*, below mentioned default parameters are used.\n
			possible parameters: \n
			[hist]: n_bins, Default: self.max_dist - self.min_dist + 1 \n
			[kde]: bwadjust, Default: 0.1 (see seaborn.kdeplot()) \n
			Default: None
		collapse: str, optional
			None if negative data should not be collapsed. ["min","max","mean","sum"] allowed as methods. See ._collapse_negative() for more information.
		ax : plt.axis
			Plot to an existing axis object. Default: None (a new axis will be created).
		color : str, optional
			Color of the plot hist/line/kde. Default: "tab:blue".
		max_dist : int, optional
			Option to set the max_dist independent of the max_dist used for counting distances. Default: None (max_dist is not changed).
		kwargs : arguments
			Additional arguments are passed to plt.hist().
		"""

		#---------------- Input checks ------------------#

		self.check_datasource("distances")
		self.check_pair(pair)
		self.check_min_max_dist()

		# check method given is supported
		tfcomb.utils.check_string(style, ["hist", "kde", "line"], name="style")

		supported_methods = ["peaks", "correction", "datasource", "distances", "scaled", "corrected", "smoothed"]
		tfcomb.utils.check_string(method, supported_methods)

		if collapse is not None:
			tfcomb.utils.check_string(collapse, ["min","max","mean","sum"])

			#collapse is not valid for other styles than hist
			if style != "hist":
				raise InputError("Setting 'collapse' is only valid for style='hist'. Please adjust input parameters.")

		# check config is dict 
		tfcomb.utils.check_type(config, [dict, type(None)], "config")

		# check save is writeable
		if save is not None:
			tfcomb.utils.check_writeability(save)
		
		# get TF1, TF2 out of pair
		tf1, tf2 = pair
		ind = tf1 + "-" + tf2 # construct index
	
		#---------------- Setup data to plot --------------#

		ylbl = "Count per distance" # standard ylbl, replace with specific option if needed
	
		#Select datasource to plot
		if method == "peaks":

			#Checking if peaks were analyzed
			if not hasattr(self, "thresh"):   #'peaks' was chosen, but signals were not yet analyzed. Falling back to signal.
				method = "signal"
				source_table = self.datasource

			else:	

				#check that pair was in zscores
				if not ind in self.thresh.index:
					self.logger.warning("TF pair '{0}' distances were counted, but not analyzed due to thresholds set in analyze_signal_all. Falling back to viewing signal.".format(ind))
					method = "signal"
					source_table = self.datasource
				
				#check which method was used for calculating peaks
				if self.thresh.loc[ind, "Method"] == "flat":
					source_table = self.datasource

				elif self.thresh.loc[ind, "Method"] == "zscore":
					source_table = self.zscores
					ylbl = "Z-score normalized counts"


		elif method == "correction":
			if style == "kde":
				raise InputError("Style 'kde' is not valid for method 'correction'. Please select another style or method.")
			self.check_datasource("uncorrected")
			source_table = self.uncorrected

		else:
			self.check_datasource(method)
			source_table = getattr(self, method)

			if method == "corrected":
				ylbl = "Corrected count per distance"
			if method == "scaled": 
				ylbl = "Scaled count per distance"

		# collapse or keep negative distances
		if collapse:
			source_table, peak_df = self._collapse_negative(source_table, method=collapse) #replaces all negative distances/peaks with "neg"
		else:
			peak_df = self.peaks

		#Establish x and y-values
		x_data = source_table.columns[2:].values #x_data is distances
		y_data = source_table.loc[ind].iloc[2:].values

		#Reduce to max_dist if chosen
		if max_dist is not None:
			x_bool = [x <= max_dist for x in x_data]
			x_data = x_data[x_bool]
			y_data = y_data[x_bool]

		#Split neg from data if neg was collapsed
		if collapse:

			#replace neg with a positional offset
			neg_pos = int(min(x_data[1:]) - int((max(x_data[1:]) - min(x_data[1:])) * 0.05)) # -5% of the full range of values

			x_neg_data = neg_pos
			y_neg_data = y_data[0]
			x_data = x_data[1:]
			y_data = y_data[1:]

		#Replace config with default parameters
		config = {} if config is None else config #initialize empty config
		default = {"bins": len(x_data), "bw_adjust": 0.1}
		for key in default:
			if key not in config:
				config[key] = default[key]

		#Check format of input config
		tfcomb.utils.check_type(config["bins"], [int], "bins")
		tfcomb.utils.check_value(config["bw_adjust"], vmin=0, name="bw_adjust")

		#-------------- Start plotting -------------#
		# start subplot
		if ax == None:
			_, ax = plt.subplots()

		#Plot signal from source using different styles
		if style == "kde": # kde plot, see seaborn.kdeplot() for more information

			if min(y_data) < 0:
				raise InputError("Style 'kde' is not valid for negative input data, e.g. if counts were corrected or if plotting zscores. Please select another method or style.") 

			sns.kdeplot(x_data, weights=y_data, bw_adjust=config["bw_adjust"], x="distance", ax=ax, color=color)

		elif style == "hist":

			#get location of bins
			_, bin_edges = np.histogram(x_data, weights=y_data, bins=config["bins"], range=(min(x_data), max(x_data)+1)) #range ensures that the last bin is included

			ax.hist(x_data, weights=y_data, bins=bin_edges-0.5, density=False, color=color, **kwargs)

			#if collapsed, plot positive and negative data separately
			if collapse:
				plt.bar(x_neg_data, y_neg_data, color='tab:orange') #plt.bar already aligns the bar to the center position, so no -0.5 adjustment is needed

			#plot axis line if y_data goes below zero (e.g. for zscore)
			if min(y_data) < 0:
				ax.axhline(0, color="tab:blue", lw=0.5)

		elif style == "line":
			ax.plot(x_data, y_data, color=color)
		

		#------------ Plot additional elements (peaks / background) ----------#

		if method == "peaks":

			thresh = self.thresh.loc[((self.thresh["TF1"] == tf1) & 
						(self.thresh["TF2"] == tf2))].iloc[0,2]

			ax.axhline(thresh, ls="--", color="grey", label="Threshold") #plot the threshold in the z-score range

		elif method == "correction":

			#Add lowess line
			lowess = self.lowess.loc[ind, x_data]
			plt.plot(x_data, lowess, color="red", label="Lowess smoothing", lw=2)
			ax.legend()

		#Show peaks in plot
		if show_peaks == True:

			#Check if peaks were calculated
			if peak_df is not None:

				#Fetch peaks and threshold line
				peak_positions = peak_df.loc[((peak_df["TF1"] == tf1) & 
											(peak_df["TF2"] == tf2))].iloc[:,2]   #peak positions in bp

				#If any peaks were found
				if len(peak_positions) > 0:

					# get indices of the peaks (mainly needed for ranges not starting with position 0)
					peak_idx = [list(x_data).index(peak) for peak in peak_positions if peak in x_data]
					x_crosses = x_data[peak_idx] #x-values for peaks
					y_crosses = y_data[peak_idx] #y-values for peaks

					# check if at least one peak is at the negative position
					if collapse:
						if "neg" in peak_positions.tolist():
							x_crosses.insert(0, x_neg_data)
							y_crosses.insert(0, y_neg_data)

					ax.plot(x_crosses, y_crosses, "x", color="red", label="Peaks") #plot peaks as crosses
					ax.legend()

		#-------- Done plotting data; making final changes to axes -------#

		# add labels to plot
		if self.percentage == True:
			ax.set_xlabel('Distance (%)')
		else:
			ax.set_xlabel('Distance (bp)')
		ax.set_ylabel(ylbl)

		#Make x-axis labels pretty
		xlim = ax.get_xlim()  #get xlim before changes
		xt = ax.get_xticks()  #positions of xticklabels
		xtl = xt.astype(int) #labels

		#Add nice x-axis label for collapsed negative counts
		if collapse:
			pos_idx = [i for i, l in enumerate(xtl) if l >= 0] 
			xt = [x_neg_data] + list(xt[pos_idx]) #x_neg_data contains the position of the neg bar
			xtl = ["neg"] + list(xtl[pos_idx])

		ax.set_xticks(xt) #explicitly set xticks to prevent matplotlib error
		ax.set_xticklabels(xtl, rotation=self._XLBL_ROTATION, fontsize=self._XLBL_FONTSIZE)
		ax.set_xlim(xlim) #set xlim back to original 

		# Final adjustments of title and spines 
		ax.set_title("{0}-{1}".format(*pair))

		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)

		#save plot
		if save is not None:
			plt.savefig(save, dpi=600, bbox_inches="tight")

		return ax
	
	def plot_network(self, color_node_by="TF1_count",
						   color_edge_by="Distance", 
						   size_edge_by="Distance_percent",
						   **kwargs): 
		"""
		Plot the rules in .rules as a network using Graphviz for python. This function is a wrapper for 
		building the network (using tfcomb.network.build_network) and subsequently plotting the network (using tfcomb.plotting.network).

		Parameters
		-----------
		color_node_by : str, optional
			A column in .rules or .TF_table to color nodes by. Default: 'TF1_count'.
		color_edge_by : str, optional
			A column in .rules to color edges by. Default: 'Distance'.
		size_edge_by : str, optional
			A column in rules to size edge width by. Default: 'TF1_TF2_count'.
		**kwargs : arguments
			All other arguments are passed to tfcomb.plotting.network.

		See also
		--------
		tfcomb.network.build_network and tfcomb.plotting.network
		"""

		#Fetch network from object or build network
		if self.network is None:
			self.logger.warning("The .network attribute is not set yet - running build_network().")
			self.build_network()			#running build network()
			
		#Plot network
		G = self.network 
		dot = tfcomb.plotting.network(G, color_node_by=color_node_by, 
										 color_edge_by=color_edge_by, 
										 size_edge_by=size_edge_by, 
										 verbosity=self.verbosity, **kwargs)

		return(dot)

