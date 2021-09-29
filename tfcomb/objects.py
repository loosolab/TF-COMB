import os 
import pandas as pd
import itertools
import datetime
import multiprocessing as mp
import numpy as np
import copy
import glob
import fnmatch
import pickle
import csv 

#Statistics
import qnorm #quantile normalization
import scipy
from scipy.stats import rankdata
from scipy.stats import norm
from scipy.stats import linregress
import statsmodels.stats.multitest
from scipy.signal import find_peaks

#Bioinfo-modules
import pysam
import pybedtools
from pybedtools import BedTool

#Modules for plotting
import matplotlib.pyplot as plt
import seaborn as sns

#Utilities from TOBIAS
import tobias
from tobias.utils.motifs import MotifList
from tobias.utils.regions import OneRegion, RegionList
from tobias.utils.utilities import merge_dicts, run_parallel, check_required
from tobias.utils.signals import fast_rolling_math

#TF-comb modules
import tfcomb
import tfcomb.plotting
import tfcomb.network
import tfcomb.analysis
from tfcomb.counting import count_co_occurrence, count_distances
from tfcomb.logging import *
from tfcomb.utils import *

from kneed import KneeLocator

class CombObj(): 
	"""
	The main class for collecting and working with co-occurring TFs.

	Examples
	----------

	>>> C = tfcomb.objects.CombObj()

	Verbosity of the output log can be set using the 'verbosity' parameter:
	>>> C = tfcomb.objects.CombObj(verbosity=2)

	"""

	#-------------------------------------------------------------------------------#
	#------------------------------- Getting started -------------------------------#
	#-------------------------------------------------------------------------------#

	def __init__(self, verbosity = 1): #set verbosity 

		#Function and run parameters
		self.verbosity = verbosity  #0: error, 1:info, 2:debug, 3:spam-debug
		self.logger = TFcombLogger(self.verbosity)
		
		#Variables for storing data
		self.prefix = None 	     #is used when objects are added to a DiffCombObj
		self.TF_names = []		 #list of TF names
		self.TF_counts = None 	 #numpy array of size n_TFs
		self.pair_counts = None	 #numpy matrix of size n_TFs x n_TFs
		self.n_bp = 0			 #predict the number of baskets 
		self.TFBS = None		 #None or filled with list of TFBS
		self.rules = None  		 #filled in by .market_basket()
		self.network = None

		#Variable for storing DistObj for distance analysis
		self.distObj = None

		#Formatted data / open files for reading
		self._genome_obj = None
		self._motifs_obj = None

	def __str__(self):
		""" Returns a """
		s = "<CombObj: "
		s += "{0} TFBS ({1} unique names)".format(len(self.TFBS), len(self.TF_names)) 

		if self.rules is not None:
			s += " | Market basket analysis: {0} rules".format(self.rules.shape[0])
		s += ">"
		return(s)

	def __repr__(self):
		return(self.__str__())
	
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
		combined.TFBS = tfcomb.utils.remove_duplicates(combined.TFBS) #remove duplicated sites 

		return(combined)
	
	def copy(self):
		""" Returns a copy of the CombObj """

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


	#-------------------------------------------------------------------------------#
	#----------------------------- Checks for the object----------------------------#
	#-------------------------------------------------------------------------------#

	def _check_TFBS(self):
		""" Internal check whether the .TFBS was already filled. Raises ValueError when .TFBS is not available"""

		#Check that TFBS exist and that it is RegionList
		if self.TFBS is None or not isinstance(self.TFBS, RegionList):
			raise ValueError("No TFBS available in '.TFBS'. The TFBS are set either using .TFBS_from_motifs, .TFBS_from_bed or TFBS_from_TOBIAS")


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
		pickle.dump(self, f_out)


	def from_pickle(self, path):
		"""
		Import a CombObj from a pickle file.

		Parameters
		-----------
		path : str
			Path to an existing pickle file to read.

		Raises
		-------
		TypeError
			If read object is not an instance of CombObj .
		
		See also
		----------
		to_pickle
		"""

		filehandler = open(path, 'rb') 
		obj = pickle.load(filehandler)

		#Check if object is CombObj
		if not isinstance(obj, CombObj):
			raise TypeError("Read object from '{0}' is not a CombObj".format(path))

		#Overwrite self with CombObj
		self = obj
		
		return(self)

	#-------------------------------------------------------------------------------#
	#-------------------------- Setting up the .TFBS list --------------------------#
	#-------------------------------------------------------------------------------#

	def TFBS_from_motifs(self, regions, motifs, genome,
								motif_pvalue = 0.0001,
								motif_naming = "name",
								gc = 0.5, 
								keep_overlaps = False, 
								threads = 1):

		"""
		Function to calculate TFBS from motifs and genome fasta.

		Parameters
		-----------
		regions : str or tobias.utils.regions.RegionList
			Path to a .bed-file containing regions or a tobias-format RegionList object. 
		motifs : str or tobias.utils.motifs.MotifList
			Path to a file containing JASPAR/MEME-style motifs or a tobias-format MotifList object.
		genome : str
			Path to the genome fasta-file to use for scan.
		motif_pvalue : float, optional
			Set the pvalue for the motif search. Default: 0.0001.
		motif_naming : str, optional
			How to name TFs based on input motifs. Must be one of "name", "". Default: "name".
		gc : float between 0-1, optional
			Set GC-content for the background model. Default: 0.5.
		keep_overlap : bool, optional
			Whether to keep overlapping occurrences of the same TFBS. Setting 'False' removes overlapping TFBS keeping the TFBS with the highest match score. Default: False.
		threads : int, optional
			How many threads to use for multiprocessing. Default: 1. 

		Returns
		-----------
		None 
			.TFBS_from_motifs fills the objects' .TFBS variable

		"""

		s = datetime.datetime.now()

		#TODO: Check input validity
		check_type(regions, [str, tobias.utils.regions.RegionList], "regions")
		check_type(motifs, [str, tobias.utils.motifs.MotifList], "motifs")
		check_type(genome, [str], "genome")
		check_type(motif_naming,[str], "motif_naming")

		#Setup regions
		if isinstance(regions, str):
			regions_f = regions
			regions = RegionList().from_bed(regions)
			self.logger.debug("Read {0} regions from {1}".format(len(regions), regions_f))

		#Setup motifs
		if isinstance(motifs, str):
			motifs_f = motifs
			motifs = tfcomb.utils.prepare_motifs(motifs_f, motif_pvalue, motif_naming)
			self.logger.debug("Read {0} motifs from '{1}'".format(len(motifs), motifs_f))
		else:
			pass
			#todo: check that given motifs have pvalues/prefix set


		#Get ready to collect TFBS
		self.TFBS = RegionList([])	#initialize empty list
		n_regions = len(regions)
		
		self.logger.info("Scanning for TFBS with {0} core(s)...".format(threads))

		#Define whether to run in multiprocessing or not
		if threads == 1:

			chunks = regions.chunks(100) 
			genome_obj = tfcomb.utils.open_genome(genome)	#open pysam fasta obj

			n_regions_processed = 0
			for region_chunk in chunks:

				#TODO: check that region_chunk is within genome_obj
				
				region_TFBS = tfcomb.utils.calculate_TFBS(region_chunk, motifs, genome_obj)
				self.TFBS += region_TFBS

				#Update progress
				n_regions_processed += len(region_chunk)
				self.logger.debug("{0:.1f}% ({1} / {2})".format(n_regions_processed/n_regions*100, n_regions_processed, n_regions))

			genome_obj.close()

		else:
			chunks = regions.chunks(threads * 2) #creates chunks of regions for multiprocessing

			#Setup pool
			pool = mp.Pool(threads)
			jobs = []
			for region_chunk in chunks:
				jobs.append(pool.apply_async(tfcomb.utils.calculate_TFBS, (region_chunk, motifs, genome, )))
			pool.close()
			
			#TODO: use log_progress 

			#Print progress
			n_prev_done = 0
			n_done = sum([job.ready() for job in jobs])
			if n_done != n_prev_done:
				self.logger.info("- Progress: {0:.2f}%".format(n_done / len(jobs) * 100))
				n_prev_done = n_done
				n_done = sum([job.ready() for job in jobs])

			results = [job.get() for job in jobs]
			pool.join()

			#Join all TFBS to one list
			self.TFBS = RegionList(sum(results, []))

		#Resolve overlaps
		if keep_overlaps == False:
			self.logger.debug("keep_overlaps == False; Resolving overlaps for TFBS")
			self.TFBS = tfcomb.utils.resolve_overlapping(self.TFBS)

		#Process TFBS
		self.TFBS.loc_sort()

		self.logger.info("Identified {0} TFBS within given regions".format(len(self.TFBS)))
		e = datetime.datetime.now()


	def TFBS_from_bed(self, bed_f):
		"""
		Fill the .TFBS attribute using a precalculated set of binding sites.

		Parameters
		-------------
		bed_f : str 
			A path to a .bed-file with precalculated binding sites.

		"""

		self.logger.info("Reading sites from '{0}'...".format(bed_f))
		self.TFBS = RegionList([OneTFBS().from_oneregion(region) for region in RegionList().from_bed(bed_f)])
		n_sites = len(self.TFBS)

		#Stats on the regions which were read
		counts = {r.name: "" for r in self.TFBS}
		n_names = len(counts)
		self.TF_names = sorted(list(set(counts.keys()))) #ensures that the same TF order is used across cores/subsets

		#Process TFBS
		self.TFBS.loc_sort()

		self.logger.info("Read {0} sites (comprising {1} unique TFs)".format(n_sites, n_names))


	def TFBS_from_TOBIAS(self, bindetect_path, condition):
		"""
		Fill the .TFBS variable with pre-calculated binding sites from TOBIAS BINDetect.

		Parameters
		-----------
		bindetect_path : str
			Path to the BINDetect-output folder containing <TF1>, <TF2>, <TF3> (...) folders.
		condition : str
			Name of condition to use for fetching bound sites.

		Raises
		-------
		ValueError 
			If no files are found in path or if condition is not one of the avaiable conditions.

		"""

		pattern = os.path.join(bindetect_path, "*", "beds", "*_bound.bed")
		files = glob.glob(pattern)

		if len(files) == 0:
			raise ValueError("No files found in path")
		
		#Check if condition given is within available_conditions
		available_conditions = set([os.path.basename(f).split("_")[-2] for f in files])

		if condition not in available_conditions:
			raise ValueError("Condition must be one of: {0}".format(list(available_conditions)))

		#Read sites from files
		condition_pattern = os.path.join(bindetect_path, "*", "beds", "*" + condition + "_bound.bed")
		condition_files = fnmatch.filter(files, condition_pattern)

		self.TFBS = RegionList()
		for f in condition_files:
			self.TFBS += RegionList([OneTFBS().from_oneregion(region) for region in RegionList().from_bed(f)])

		#Process TFBS
		self.TFBS.loc_sort()

		self.logger.info("Read {0} sites from condition '{1}'".format(len(self.TFBS), condition))
		
	#-------------------------------------------------------------------------------------------------------------#
	#----------------------------------- Filtering and processing of TFBS ----------------------------------------#
	#-------------------------------------------------------------------------------------------------------------#

	def cluster_TFBS(self, threshold=0.5):
		""" 
		Cluster TFBS based on overlap of individual binding sites. This can be used to pre-process TFBS into TF "families" of TFs with similar motifs.
		
		Parameters
		------------
		threshold : float from 0-1, optional
			The threshold to set when clustering binding sites. Default: 0.5
		
		"""

		self._check_TFBS()

		#todo: Test input threshold		

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

		#Print out new names
		self.logger.info("")


	def subset_TFBS(self, regions):
		"""
		Subset .TFBS in object to specific regions. Can be used to select only a subset of TFBS (e.g. only in promoters) to run analysis on.

		Parameters
		-----------
		regions : str or RegionList
			Path to a .bed-file containing regions or a tobias-format RegionList object. 

		Returns
		-------
		None
			The .TFBS attribute is updated in place

		"""

		self._check_TFBS()
		tfcomb.utils.check_type(regions, [str, tobias.utils.regions.RegionList], "regions")

		#If regions are string, read to internal format
		if isinstance(regions, str):
			regions = RegionList().from_bed(regions)
		
		#Create regions->sites dict
		TFBS_in_regions = tfcomb.utils.assign_sites_to_regions(self.TFBS, regions)

		#Merge across keys
		self.TFBS = RegionList(sum([TFBS_in_regions[key] for key in TFBS_in_regions], []))
		self.TFBS.loc_sort()

		self.logger("The sites found in .TFBS were subset to the regions given.")

	def TFBS_to_bed(self, path):
		"""
		Writes out the .TFBS regions to a file. This is a wrapper for the tobias.utils.regions.RegionList().write_bed() utility.

		Parameters
		----------
		path : str
			File path to write .bed-file to.
		"""
		self._check_TFBS()

		#check_writeability(path) #TODO: Check if bed is writeable
		self.TFBS.write_bed(path)


	#-------------------------------------------------------------------------------------------------------------#
	#----------------------------------------- Counting co-occurrences -------------------------------------------#
	#-------------------------------------------------------------------------------------------------------------#

	def count_within(self, min_distance = 0, 
						   max_distance = 100, 
						   max_overlap = 0, 
						   stranded = False, 
						   directional = False, 
						   binarize = False,
						   anchor = "inner"):
		""" 
		Count co-occurrences between TFBS. This function requires .TFBS to be filled by either `TFBS_from_motifs`, `TFBS_from_bed` or `TFBS_from_tobias`. 
		This function can be followed by .market_basket to calculate association rules.
		
		Parameters
		-----------
		min_distance : int
			Minimum distance between two TFBS to be counted as co-occurring. Distances are calculated from end-to-end of regions, e.g. the smallest possible distance. Default: 0.
		max_distance : int
			Maximum distance between two TFBS to be counted as co-occurring. Distances are calculated from end-to-end of regions, e.g. the smallest possible distance. Default: 100.
		max_overlap : float between 0-1
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
			inner/mid/outer

		Returns
		----------
		None 
			Fills the object variables .TF_counts and .pair_counts.
		
		Raises
		--------
		ValueError
			If .TFBS has not been filled.
		"""


		self._check_TFBS()

		
			
		self.logger.info("Counting co-occurring TFs from .TFBS...")

		#Should strand be taken into account?
		TFBS = copy.deepcopy(self.TFBS)
		if stranded == True:
			for site in TFBS:
				site.name = "{0}({1})".format(site.name, site.strand)

		#Find all names within TFBS 
		self.TF_names = sorted(list(set([site.name for site in TFBS]))) #ensures that the same TF order is used across cores/subsets
		n_TFs = len(self.TF_names)
		self.logger.debug("Found {0} TF names in TFBS".format(n_TFs))
		#self.logger.spam("TFBS names: {0}".format(self.TF_names))

		#Convert TFBS to internal numpy integer format
		chromosomes = {site.chrom:"" for site in TFBS}.keys()
		chrom_to_idx = {chrom: idx for idx, chrom in enumerate(chromosomes)}
		name_to_idx = {name: idx for idx, name in enumerate(self.TF_names)}
		sites = np.array([(chrom_to_idx[site.chrom], site.start, site.end, name_to_idx[site.name]) for site in TFBS]) #numpy integer array
	
		#Count number of bp covered by all TFBS
		self.TFBS_bp = len(self.TFBS) #get_unique_bp(self.TFBS)

		#---------- Count co-occurrences within TFBS ---------#

		self.logger.debug("Counting co-occurrences within sites")
		binary = 0 if binarize == False else 1
		TF_counts, pair_counts = count_co_occurrence(sites, min_distance,
															max_distance,
															max_overlap, 
															binary, 
															n_TFs)
		pair_counts = tfcomb.utils.make_symmetric(pair_counts) if directional == False else pair_counts	#Deal with directionality

		self.TF_counts = TF_counts
		self.pair_counts = pair_counts

		#Update object variables
		self.rules = None 	#Remove .rules if market_basket() was previously run
		self.min_distance = min_distance
		self.max_distance = max_distance
		self.stranded = stranded
		self.directional = directional
		self.max_overlap = max_overlap

		self.logger.info("Done finding co-occurrences! Run .market_basket() to estimate significant pairs")


	def get_pair_locations(self, TF1, TF2, TF1_strand = None,
										   TF2_strand = None,
										   min_distance = 0, 
										   max_distance = 100, 
										   max_overlap = 0,
										   directional = False):
		""" Get genomic locations of a particular TF pair. Requires .TFBS to be filled.
		
		Parameters
		----------
		TF1 : str 
			Name of TF1 in pair.
		TF2 : str 
			Name of TF2 in pair.
		TF1_strand : str
			Strand of TF1 in pair. Default: None (strand is not taken into account).
		TF2_strand : str
			Strand of TF2 in pair. Default: None (strand is not taken into account).
		min_distance : int
			Default: 0
		max_distance : int
			Default: 100
		max_overlap : float
			Default: 0
		directional : bool
			Default: False

		Returns
		-------
		List of tuples in the form of: [(OneRegion, OneRegion, distance), (...)]
			Each entry in the list is a tuple of OneRegion() objects giving the locations of TF1/TF2 + the distance between the two regions

		See also
		---------
		count_within

		"""

		self._check_TFBS()

		locations = [] #empty list of regions

		TF1_tup = (TF1, TF1_strand)
		TF2_tup = (TF2, TF2_strand)
		sites = self.TFBS
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
						
							#True if these TFBS co-occur within window
							distance = TF2_start - TF1_end
							distance = 0 if distance < 0 else distance

							if TF1_chr == TF2_chr and (distance <= max_distance):

								if distance >= min_distance:
								
									# check if they are overlapping more than the threshold
									valid_pair = 1
									if distance == 0:
										overlap_bp = TF1_end - TF2_start
										
										# Get the length of the shorter TF
										short_bp = min([TF1_end - TF1_start, TF2_end - TF2_start])
										
										#Invalid pair, overlap is higher than threshold
										if overlap_bp / (short_bp*1.0) > max_overlap: 
											valid_pair = 0

									#Save association
									if valid_pair == 1:

										#Save location
										reg1 = OneTFBS(chrom=TF1_chr, start=TF1_start, end=TF1_end, name=TF1_name, strand=TF1_strand_i)
										reg2 = OneTFBS(chrom=TF2_chr, start=TF2_start, end=TF2_end, name=TF2_name, strand=TF2_strand_i)
										location = (reg1, reg2, distance)
										locations.append(location)

							else:
								#The next site is out of window range; increment to next i
								i += 1
								finding_assoc = False   #break out of finding_assoc-loop
			
			else: #current TF1 is not TF1/TF2; go to next site
				i += 1

		return(locations)

	#-----------------------------------------------------------------------------------------#
	#-------------------------------- Market basket analysis ---------------------------------#
	#-----------------------------------------------------------------------------------------#

	def market_basket(self, measure="cosine", threads = 1):
		"""
		Runs market basket analysis on the TF1-TF2 counts. Requires prior run of .count_within().
	
		Parameters
		-----------
		measure : str or list of strings, optional
			The measure(s) to use for market basket analysis. Can be any of: ["cosine", "confidence", "lift", "jaccard"]. Default: 'cosine'.
		threads : int, optional
			Threads to use for multiprocessing. Default: 1.

		Raises
		-------
		ValueError 
			If no TF counts are available.
		"""

		available_measures = ["cosine", "confidence", "lift", "jaccard"]

		#Check that TF counts are available
		if (self.TF_counts is None) or (self.pair_counts is None):
			raise ValueError("No counts available. Please run either .count_within() or .count_between()")

		##### Calculate market basket analysis #####
		#Estimate the number of baskets
		n_baskets = self.TFBS_bp

		#Convert pair counts to table and convert to long format
		pair_counts_table = pd.DataFrame(self.pair_counts, index=self.TF_names, columns=self.TF_names) #size n x n TFs
		pair_counts_table["TF1"] = pair_counts_table.index
		table = pd.melt(pair_counts_table, id_vars=["TF1"], var_name=["TF2"], value_name="TF1_TF2_count")  #long format (TF1, TF2, value)

		#Add TF single counts to table
		vals = zip(self.TF_names, self.TF_counts)
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
				raise ValueError("Measure '{0}' is invalid. The measure must be one of: {1}".format(metric, available_measures))
		
		#Remove rows with TF1_TF2_count == 0
		table = table[table["TF1_TF2_count"] != 0]

		#Sort for highest measure pairs
		table.sort_values(measure, ascending=False, inplace=True)
		table.reset_index(inplace=True, drop=True)

		#Calculate p-values for the measure(s) given
		self.logger.debug("Calculating p-value for {0} rules".format(len(table)))
		if threads == 1:
			self.logger.info("Parameter 'threads' is set to '1' - to speed up p-value calculation, please increase the number of threads used.")
		
		for metric in measure:
			self.logger.debug("Calculating p-value for {0}".format(metric))
			tfcomb.utils.tfcomb_pvalue(table, measure=metric, threads=threads, logger=self.logger) #adds pvalue column to table

		#Create internal node table for future network analysis
		TF1_table = table[["TF1", "TF1_count", "TF1_support"]].set_index("TF1", drop=False).drop_duplicates()
		TF2_table = table[["TF2", "TF2_count", "TF2_support"]].set_index("TF2", drop=False).drop_duplicates()
		self.TF_table = TF1_table.merge(TF2_table, left_index=True, right_index=True)

		#Set name of index for table
		table.index = table["TF1"] + "-" + table["TF2"]

		#Market basket is done; save to .rules
		self.logger.info("Market basket analysis is done! Results are found in <CombObj>.rules")
		self.rules = table

	#-----------------------------------------------------------------------------------------#
	#------------------------------ Selecting significant rules ------------------------------#
	#-----------------------------------------------------------------------------------------#

	def simplify_rules(self):
		""" Simplify rules so that TF1-TF2 and TF2-TF1 pairs only occur once within .rules. 
		This is useful for association metrics such as 'cosine', where the association of TF1->TF2 equals TF2->TF1. 
		This function keeps the first unique pair occurring within the rules table. """

		#TODO:check that rules are present

		
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
		
	def select_TF_rules(self, TF_list, TF1=True, TF2=True):
		""" Select rules based on a list of TF names. 
		
		Parameters
		------------
		TF_list : list
			List of TF names fitting to TF1/TF2 within .rules.
		TF1 : bool
			Whether to subset the rules containing 'TF_list' TFs within "TF1". Default: True.
		TF2 : bool
			Whether to subset the rules containing 'TF_list' TFs within "TF2". Default: True.
		"""

		selected = self.rules.copy()

		if TF1 == True:
			selected_bool = selected["TF1"].isin(TF_list)
			if sum(selected_bool) == 0:
				self.logger.warning("")
			else:
				selected = selected[selected_bool]
		
		if TF2 == True:
			selected_bool = selected["TF2"].isin(TF_list)
			if sum(selected_bool) == 0:
				self.logger.warning("")
			else:
				selected = selected[selected_bool]

		#Create new object with selected rules
		new_obj = self.copy()
		new_obj.rules = selected

		selected_names = list(set(selected["TF1"].tolist() + selected["TF2"].tolist()))
		new_obj.TFBS = RegionList([site for site in self.TFBS if site.name in selected_names])

		return(new_obj)

	def select_top_rules(self, n):
		"""
		Select the top 'n' rules within .rules. By default, the .rules are sorted for the measure value, so n=100 will select the top 100 highest values for the measure (e.g. cosine).

		Parameters
		-----------
		n : int
			The number of rules to select.

		Returns
		--------
		tfcomb.objects.CombObj()
			An object containing a subset of <obj>.rules
		"""

		#Check input types
		tfcomb.utils.check_type(n, [int])

		#Select top n_rules from .rules
		selected = self.rules.copy()
		selected = selected[:n]

		#Create new object with selected rules
		new_obj = self.copy()
		new_obj.rules = selected

		selected_names = list(set(selected["TF1"].tolist() + selected["TF2"].tolist()))
		new_obj.TFBS = RegionList([site for site in self.TFBS if site.name in selected_names])

		return(new_obj)

	def select_significant_rules(self, measure="cosine", 
										pvalue="cosine_pvalue", 
										measure_threshold=None,
										pvalue_threshold=0.05,
										plot=True, 
										**kwargs):
		"""
		Make selection of rules based on distribution of measure and pvalue.

		Parameters
		-----------
		measure : str, optional
			The name of the column within .rules containing the measure to be selected on. Default: 'cosine'.
		pvalue : str, optional
			The name of the column within .rules containing the pvalue to be selected on. Default: 'cosine_pvalue'
		measure_threshold : float, optional
			A minimum threshold for the measure to be selected. If None, the threshold will be estimated through a knee-plot of the cumulative sum of values. Default: None.
		pvalue_threshold : float, optional
			A p-value threshold for selecting rules. Default: 0.05.
		plot : bool, optional
			Whether to show the 'measure vs. pvalue'-plot or not. Default: True.
		kwargs : arguments
			Additional arguments are forwarded to tfcomb.plotting.volcano

		Returns
		--------
		tfcomb.objects.CombObj()
			An object containing a subset of <obj>.rules

		See also
		---------
		tfcomb.plotting.volcano
		"""

		#Check if measure are in columns
		if measure not in self.rules.columns:
			raise KeyError("Measure column '{0}' is not in .rules".format(measure))

		#If pvalue not in columns; calculate pvalue for measure
		if pvalue not in self.rules.columns:
			self.logger.warning("pvalue column given ('{0}') is not in .rules".format(pvalue))
			self.logger.warning("Calculating pvalues from measure '{0}'".format(measure))

			self.calculate_pvalues(measure=measure)
			pvalue = measure + "_pvalue"

		#If measure_threshold is None; try to calculate optimal threshold via knee-plot
		if measure_threshold == None:
			self.logger.info("measure_threshold is None; trying to calculate optimal threshold")
			
			#Compute distribution histogram of measure values
			y, x = np.histogram(self.rules[measure], bins=100)
			x = [np.mean([x[i], x[i+1]]) for i in range(len(x)-1)] #Get mid of start/end of each bin
			y = np.cumsum(y)
			kneedle = KneeLocator(x, y, curve="concave", direction="increasing", interp_method="polynomial")
			measure_threshold = kneedle.knee

		#Set threshold on table
		selected = self.rules.copy()
		selected = selected[(selected[measure] >= measure_threshold) & (selected[pvalue] <= pvalue_threshold)]

		if plot == True:
			tfcomb.plotting.volcano(self.rules, measure = measure, 
												pvalue = pvalue, 
												measure_threshold = measure_threshold,
												pvalue_threshold = pvalue_threshold,
												**kwargs)

		#Create a CombObj with the subset of TFBS and rules
		self.logger.info("Creating subset of TFBS and rules using thresholds")
		new_obj = self.copy()
		new_obj.rules = selected

		selected_names = list(set(selected["TF1"].tolist() + selected["TF2"].tolist()))
		new_obj.TFBS = RegionList([site for site in self.TFBS if site.name in selected_names])

		return(new_obj)


	#-----------------------------------------------------------------------------------------#
	#-------------------------------- Plotting functionality  --------------------------------#
	#-----------------------------------------------------------------------------------------#

	def plot_background(self, TF1, TF2):
		pass



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

		#Sort rules
		table = self.rules.copy()
		if sort_by is not None:
			table = table.sort_values(sort_by, ascending=False)

		#Select n top rules
		top_rules = table.head(n_rules)
		top_rules.index = top_rules["TF1"].values + " + " + top_rules["TF2"].values

		#Plot
		ax = tfcomb.plotting.bubble(top_rules, yaxis=yaxis, color_by=color_by, size_by=size_by, **kwargs)

	#-------------------------------------------------------------------------------------------#
	#----------------------------------- In-depth analysis -------------------------------------#
	#-------------------------------------------------------------------------------------------#

	def create_distObj(self):
		""" Creates a distObject, useful for manual analysis. 
			 Fills self.distObj.
		"""
		#TODO: check rules filled
		self.distObj = DistObj()
		self.distObj.fill_rules(self)
		self.distObj.logger.info("DistObject successfully created! It can be accessed via combobj.distObj")

	def analyze_distances(self,normalize = True, n_bins = None, parent_directory = None, plot_signal = True, **kwargs):
		""" Standard distance analysis workflow.
			Use create_distObj for own workflow steps and more options!
		"""
		self.create_distObj()
		self.distObj.count_distances(normalize=normalize,directional = self.directional)
		# TODO: check parent_directory and create nice subfolder structure !
		self.distObj.linregress_all(n_bins = n_bins, save= parent_directory)
		self.distObj.correct_all(n_bins = n_bins, save= parent_directory)
		self.distObj.analyze_signal_all(**kwargs,save= parent_directory)
		self.distObj.logger.info("Results are found in CombObj.distObj.peaks")

	def bed_from_range(self, TF1, TF2, TF1_strand = None,
									   TF2_strand = None,
									   directional = False,
									   dist_range = None,
									   save = None,
									   delim = "\t"):
		""" Creates a bed file ("chr","pos start","pos end","name TF1", "strand","chr","pos start","pos end","name TF2", "strand","distance")
			for a given TF-pair. Optional a range can be specified e.g. dist_range = (30,40) gives all hist with distances between 30 and 40

			Parameters
			----------
			TF1 : str 
				Name of TF1 in pair.
			TF2 : str 
				Name of TF2 in pair.
			TF1_strand : str
				Strand of TF1 in pair. Default: None (strand is not taken into account).
			TF2_strand : str
				Strand of TF2 in pair. Default: None (strand is not taken into account).
			directional : bool
				Default: False
			dist_range: tuple
				Range start and end to save e.g. (30,40). Default: None (write all ranges)
			save:
				Output Path to write results to. (filename will be constructed automatically from TF1-/TF2-name)
				Default: None (results will not be saved)

				
			Returns
			-------
			List of tuples in the form of: [(OneRegion, OneRegion, distance), (...)]
				Each entry in the list is a tuple of OneRegion() objects giving the locations of TF1/TF2 + the distance between the two regions

		"""
		max_over = 0
		if self.min_dist < 0: 
			max_over = -self.min_dist
		
		b = self.get_pair_locations(TF1, TF2, TF1_strand = TF1_strand,
										   TF2_strand = TF2_strand,
										   min_distance = self.min_dist, 
										   max_distance = self.max_dist, 
										   max_overlap = max_over,
										   directional = directional)
		
		if save is not None:
			# TODO: Check if save is a valid path
			with open(f'{save}{TF1}_{TF2}.csv',"w") as outfile :
				header_row = ["chr","pos start","pos end","name TF1", "strand","chr","pos start","pos end","name TF2", "strand","distance"]
				csv_file = csv.writer(outfile,delimiter=delim) 
				csv_file.writerow(header_row) 
				for line in b:
					tf1_region = line[0]
					tf2_region = line[1]
					dist = line[2]
					if dist_range is not None:
						if (dist in range(dist_range[0],dist_range[1])):
							content = [tf1_region.chrom,tf1_region.start,tf1_region.end,tf1_region.name,tf1_region.strand,
									   tf2_region.chrom,tf2_region.start,tf2_region.end,tf2_region.name,tf2_region.strand,
									   dist]
							csv_file.writerow(content)
						else:
							content = [tf1_region.chrom,tf1_region.start,tf1_region.end,tf1_region.name,tf1_region.strand,
								   tf2_region.chrom,tf2_region.start,tf2_region.end,tf2_region.name,tf2_region.strand,
								   dist]
						csv_file.writerow(content) 
		return b


	#-------------------------------------------------------------------------------------------#
	#------------------------------------ Network analysis -------------------------------------#
	#-------------------------------------------------------------------------------------------#

	def build_network(self):
		""" Builds a TF-TF co-occurrence network for the rules within object. This is a wrapper for the tfcomb.network.build_nx_network() function, 
			 which uses the python networkx package. 
			 
		Returns
		-------
		None - fills the .network attribute of the `CombObj` with a networkx.Graph object
		"""

		#Build network
		self.logger.debug("Building network using tfcomb.network.build_nx_network")
		self.network = tfcomb.network.build_nx_network(self.rules, node_table=self.TF_table, verbosity=self.verbosity)
		

	def partition_network(self, method="louvain", weight=None):
		"""
		Creates a partition of nodes within network
		
		Note: Requires build_network 
		Adds a new node attribute 'partition' to network

		Parameters
		-----------
		method : str, one of ["louvain", "blockmodel"]
			Default: "louvain".

		"""
		#Fetch network from object
		if self.network is None:
			self.logger.info("The .network attribute is not available - running .build_network()")
			self.build_network()
			
		#Decide method of partitioning
		if method == "louvain":
			tfcomb.network.partition_louvain(self.network, weight=weight, logger=self.logger) #this adds "partition" to the network
			self.logger.info("Added 'partition' attribute to the network attributes")

			node_table = tfcomb.network.get_node_table(self.network)

		elif method == "blockmodel":

			#Create gt network	
			self._gt_network = tfcomb.network.build_gt_network(self.rules, node_table=self.TF_table, verbosity=self.verbosity)

			#Partition network
			tfcomb.network.partition_blockmodel(self._gt_network)

			node_table = tfcomb.network.get_node_table(self._gt_network)
			node_table.set_index("TF1", drop=False, inplace=True)

		else:
			raise ValueError("Method must be one of: ['louvain', 'blockmodel']")
			
		#Update TF_table
		self.logger.debug("TF_table: {0}".format(node_table.head(5)))
		self.TF_table = node_table
		
		#Update network attribute for plotting
		if method == "blockmodel":
			self.network = tfcomb.network.build_nx_network(self.rules, node_table=self.TF_table, verbosity=self.verbosity)
	
	def plot_network(self, color_node_by="TF1_count",
						   color_edge_by="cosine", 
						   size_edge_by="TF1_TF2_count",
						   **kwargs): 
		"""
		Plot the rules in .rules as a network using Graphviz for python. This function is a wrapper for 
		building the network (using tfcomb.network.build_network) and subsequently plotting the network (using tfcomb.plotting.network).
		Requires build_network() to be run.

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

	def compare(self, obj_to_compare, measure="cosine", normalize=True):
		"""
		Utility function to create a DiffCombObj directly from a comparison between self and another CombObj. Requires .market_basket() run on both objects.
		Runs DiffCombObj.normalize (if chosen), DiffCombObj.calculate_foldchanges() under the hood. 

		Note
		------
		Set .prefix for each object to get proper naming of output log2fc columns. 

		Parameters
		---------
		obj_to_compare : tfcomb.objects.CombObj
			Another CombObj to compare to the 
		measure : str
			The measure to compare between objects. Default: 'cosine'.
		normalize : bool
			Whether to normalize values between objects

		Return
		-------
		DiffCombObj
		"""
		
		#TODO: Check that market basket was run on both objects

		
		diff = DiffCombObj([self, obj_to_compare], verbosity=self.verbosity)

		if normalize == True:
			diff.normalize()

		diff.calculate_foldchanges() #also calculates p-values

		return(diff)



###################################################################################
############################## Differential analysis ##############################
###################################################################################


class DiffCombObj():

	def __init__(self, objects = [], measure='cosine', verbosity=1):
		""" Initializes a DiffCombObj object for doing differential analysis between CombObj's.

		Parameters
		----------
		objects : list, optional
			A list of CombObj instances. If list is empty, an DiffCombObj will be created. Default: [].
		measure : str, optional
			The measure to compare between objects. Must be a column within .rules for each object. Default: 'cosine'
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
			self.add_object(obj)

	def __str__(self):
		pass

	def add_object(self, obj):
		"""
		Add one CombObj to the DiffCombObj.

		Parameters
		-----------
		obj : CombObj
			An instance of CombObj
		"""

		#TODO: Check that object is an instance of CombObj
		check_type(obj, [CombObj])

		#TODO: check that prefixes are unique; otherwise, throw error F
		#Check if prefix is set - otherwise, set to obj<int>
		if obj.prefix is not None:
			prefix = obj.prefix
		else:
			prefix = "Obj" + str(self.n_objects + 1)
			#logger warning

		#TODO:
		#check that all objects contain self.measure

		#Format table from obj to contain TF1/TF2 + measures with prefix
		columns_to_keep = ["TF1", "TF2"] + [self.measure]
		obj_table = obj.rules[columns_to_keep] #only keep necessary columns
		obj_table.rename(columns={self.measure: str(prefix) + "_" + self.measure}, inplace=True)

		#Initialize table if this is the first object
		if self.n_objects == 0: 
			self.rules = obj_table

		#Or add object to this DiffCombObj
		else:
			self.rules = self.rules.merge(obj_table, left_on=["TF1", "TF2"], right_on=["TF1", "TF2"], how="outer")
			self.rules = self.rules.fillna(0) #Fill NA with null (happens if TF1/TF2 pairs are different between objects)

		#TODO: ensure the same TFBS were present in all objects
		
		self.n_objects += 1 #current number of objects +1 for the one just added
		self.prefixes.append(prefix)
		

	def normalize(self):
		"""
		Normalize the values for the given measure (.measure) using quantile normalization. 
		Overwrites the <prefix>_<measure> columns in .rules with the normalized values.
		"""

		#Establish input/output columns
		measure_columns = [prefix + "_" + self.measure for prefix in self.prefixes]
		
		#Normalize values
		self.rules[measure_columns] = qnorm.quantile_normalize(self.rules[measure_columns], axis=1)

		#TODO: Ensure that original 0 values are kept at 0


	def calculate_foldchanges(self, pseudo = None):
		""" Calculate measure foldchanges and p-values between objects in DiffCombObj. The measure is chosen at the creation of the DiffCombObj and defaults to 'cosine'.
		
		Parameters
		----------
		pseudo : float, optional
			Set the pseudocount to add to all values before log2-foldchange transformation. Default: None (pseudocount will be estimated per contrast).
		threads : int, optional
			The number of threads to use for calculating foldchanges and p-values

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
			self.logger.info("Calculating contrast: {0} / {1}".format(p1, p2))
			log2_col = "{0}/{1}_{2}_log2fc".format(p1, p2, measure)
			columns.append(log2_col)

			p1_values = self.rules[p1 + "_" + measure]
			p2_values = self.rules[p2 + "_" + measure]

			#Estimate pseudocount
			if pseudo == None:
				vals = self.rules[[p1 + "_" + measure, p2 + "_" + measure]].values.ravel()
				pseudo = np.percentile(vals[vals>0], 25) #25th percentile of values >0
				self.logger.debug("Pseudocount: {0}".format(pseudo))

			self.rules[log2_col] = np.log2((p1_values + pseudo) / (p2_values + pseudo))

			#Calculate p-value of each pair
			self.logger.debug("Calculating p-value")
			pvalue_col = "{0}/{1}_{2}_pvalue".format(p1, p2, measure)
			self.rules[pvalue_col] = tfcomb.utils._calculate_pvalue(self.rules, measure=log2_col, alternative="two-sided")

		#Sort by first contrast log2fc
		self.logger.debug("columns: {0}".format(columns))
		self.rules.sort_values(columns[0], inplace=True)

		self.logger.info("Please find the calculated log2fc's in the rules table (<DiffCombObj>.rules)")
		

	def select_n_rules():
		pass


	def select_rules(self, contrast=None,
						   measure="cosine", 
						   measure_threshold=None,
						   pvalue_threshold=0.05,
						   plot = True, 
						   **kwargs):
		"""
		Select differentially regulated rules on the basis of measure and pvalue.

		Parameters
		-----------
		contrast : tuple
			Name of the contrast to use in tuple format e.g. (<prefix1>,<prefix2>). Default: None (the first contrast is shown).
		measure : str
			The measure to use for selecting rules. Default: "cosine".
		measure_threshold : tuple
			Default: 
		pvalue_threshold : float
			Default: 
		plot : boolean


		See also
		----------
		tfcomb.plotting.volcano
		"""
		
		#Identify measure to use based on contrast
		if contrast == None:
			contrast = self.contrasts[0]

		self.logger.info("Selecting rules for contrast: {0}".format(self.contrasts[0]))
		measure_col = "{0}/{1}_{2}_log2fc".format(contrast[0], contrast[1], measure)
		self.logger.debug("Measure column is: {0}".format(measure_col))

		#Calculate pvalue
		pvalue_col = "{0}/{1}_{2}_pvalue".format(contrast[0], contrast[1], measure)
		if pvalue_col not in self.rules.columns:
			self.logger.warning("pvalue column given ('{0}') is not in .rules".format(pvalue_vol))
			self.logger.warning("Calculating pvalues from measure '{0}'".format(measure_col))

			self.rules[measure + "_pvalue"] = tfcomb.utils._calculate_pvalue(self.rules, measure=measure_col)
	
		#Find optimal measure threshold
		self.logger.info("measure_threshold is None; trying to calculate optimal threshold")

		#TODO: Assume that the log2fc background is normal


		if plot == True:
			tfcomb.plotting.volcano(self.rules, measure=measure_col, pvalue=pvalue_col, **kwargs)
		

		#Create a CombObj with the subset of TFBS and rules
		self.logger.info("Creating subset of TFBS and rules using thresholds")
		new_obj = self.copy()
		new_obj.rules = selected

		selected_names = list(set(selected["TF1"].tolist() + selected["TF2"].tolist()))
		new_obj.TFBS = [site for site in self.TFBS if site.name in selected_names]

		return(new_obj)

	#-------------------------------------------------------------------------------------------#
	#----------------------------- Plots for differential analysis -----------------------------#
	#-------------------------------------------------------------------------------------------#

	def plot_correlation(self, method="pearson"):
		"""
		Plot correlation between rules across objects.

		Parameters
		-----------
		method : str
			Either 'pearson' or 'spearman'. Default: 'pearson'.
		"""

		#Define columns
		cols = [prefix + "_" + self.measure for prefix in self.prefixes]

		#Calculate matrix and plot
		matrix = self.rules[cols].corr(method=method)
		sns.clustermap(matrix,
							cbar_kws={'label': method})

	def plot_heatmap(self, contrast=None, 
						   n_rules=10, 
						   color_by="cosine_log2fc", 
						   sort_by=None, 
						   **kwargs):
		"""
		Functionality to plot a heatmap of differentially co-occurring TF pairs. 

		Parameters
		------------
		contrast : tuple
			Name of the contrast to use in tuple format e.g. (<prefix1>,<prefix2>). Default: None (the first contrast is shown).
		n_rules : int
			Number of rules to show from each contrast (default: 10). Note: This is the number of rules either up/down, meaning that the rules shown are n_rules * 2.
		color_by : str
			Default: "cosine" (converted to "<prefix1>/<prefix2>_<color_by>")
		sort_by : str
			Default: None (keep sort)
		kwargs : arguments
			Additional arguments are passed to tfcomb.plotting.heatmap.

		See also
		----------
		tfcomb.plotting.heatmap
		"""

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
						  color_by="lift_log2fc", 
						  size_by="lift_log2fc", 
						  **kwargs):
		"""
		Plot bubble scatterplot of information within .rules.

		Parameters
		-----------
		contrast : tuple
			Name of the contrast to use in tuple format e.g. (<prefix1>,<prefix2>). Default: None (the first contrast is shown).
		n_rules : int
			Number of rules to show (in each direction). Default: 20.
		yaxis : str
			Measure to show on the y-axis. Default: "cosine_log2fc".
		color_by : str
			If column is not in rules, the string is supposed to be in the form "prefix1/prefix2_<color_by>".

		size_by : str

		kwargs : arguments

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

		tfcomb.plotting.bubble(data=top_rules, yaxis=yaxis, color_by=color_by, size_by=size_by, ax=ax1)
		tfcomb.plotting.bubble(data=bottom_rules, yaxis=yaxis, color_by=color_by, size_by=size_by, ax=ax2)

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
			Column in .rules to size_node_by. 
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
		G = tfcomb.network.build_nx_network(selected)
		
		#Plot network
		self.logger.debug("Plotting network using 'tfcomb.plotting.network'")
		dot = tfcomb.plotting.network(G, color_node_by=color_node_by, size_node_by=size_node_by, 
										 color_edge_by=color_edge_by, size_edge_by=size_edge_by, 
										 verbosity = self.verbosity, **kwargs)

		return(dot)

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

	def __init__(self, verbosity = 1): #set verbosity 

		#Function and run parameters
		self.verbosity = verbosity  #0: error, 1:info, 2:debug, 3:spam-debug
		self.logger = TFcombLogger(self.verbosity)
		
		#Variables for storing data
		self.rules = None  		     # Filled in by .fill_rules()
		self.TF_names = []		     # List of TF names
		self._raw = None             # Raw distance data [Numpy array of size n_pairs x maxDist]

		self.distances = None 	     # Pandas DataFrame of size n_pairs x maxDist
		self.corrected = None        # Pandas DataFrame of size n_pairs x maxDist
		self.linres = None           # Pandas DataFrame of size n_pairs x maxDist
		self.normalized = None       # Pandas DataFrame of size n_pairs x maxDist
		self.smoothed = None         # Pandas DataFrame of size n_pairs x maxDist
		self.peaks = None 	         # Pandas DataFrame of size n_pairs x n_preferredDistance 

		self.peaking_count = None    # Number of pairs with at least one peak 
	
		
		self.n_bp = 0			     # Predicted number of baskets 
		self.TFBS = RegionList()     # None RegionList() of TFBS
		self.smooth_window = 3       # Smoothing window size, 1 = no smoothing
		self.anchor_mode = 0         # Distance measure mode [0,1,2]

		# str <-> int encoding dicts
		self.name_to_idx = None      # Mapping TF-names: string <-> int 
		self.pair_to_idx = None      # Mapping Pairs: tuple(string) <-> int

		# analysis parameters
		self.min_dist = 0            # Minimum distance. Default: 0 
		self.max_dist = 300          # Maximum distance. Default: 300
		self.max_overlap = 0         # Maximum overlap. Default: 0       
		self.directional = None      # True if direction is taken into account, false otherwise 
   
		self._PEAK_HEADER = "TF1\tTF2\tDistance\tPeak Heights\tProminences\tProminence Threshold\n"

	def __str__(self):
		pass
	
	def set_verbosity(self, level):
		""" Set the verbosity level for logging after creating the CombObj.

		Parameters
		----------
		level : int
			A value between 0-3 where 0 (only errors), 1 (info), 2 (debug), 3 (spam debug). 
		
		Returns
		----------
		None 
			Sets the verbosity level for the Logger inplace
		"""

		self.verbosity = level
		self.logger = TFcombLogger(self.verbosity) #restart logger with new verbosity	    
	
	def fill_rules(self,comb_obj):
		""" Fill DistanceObject according to reference object with all needed Values and parameters
		to perform standard prefered distance analysis

		Parameters
		----------
		comb_obj: tfcomb.objects
			Object from which the rules and parameters should be copied from

		Returns
		----------
		None 
			Copies values and parameters from a combObj or diffCombObj.
		
		"""

	   
		# TODO: Check Rules 
		
		#copy rules
		self.rules = comb_obj.rules
		# reset pandas index
		self.rules = self.rules.reset_index(drop=True)

		# copy parameters
		self.TF_names = comb_obj.TF_names
		self.TFBS = comb_obj.TFBS 
		self.min_dist = comb_obj.min_distance
		self.max_dist = comb_obj.max_distance
		self.directional = comb_obj.directional
		self.max_overlap = comb_obj.max_overlap
		#self.anchor = comb_obj.anchor

	def set_anchor(self,anchor):
		""" set anchor for distance measure mode
		0 = inner
		1 = outer
		2 = center

		Parameters
		----------
		anchor : str or int
			one of ["inner","outer","center"] or [0,1,2]

		Returns
		----------
		None 
			Sets anchor mode inplace
		"""

		try:
			tfcomb.utils.check_type(anchor,[str,int],"anchor")
		except ValueError as e:
			self.logger.error(str(e))
			sys.exit(0)

		if isinstance(anchor,str):
			try:
				tfcomb.utils.check_string(anchor,["inner","outer","center"])
			except ValueError as e:
				self.logger.error(str(e))
				sys.exit(0)

			modes = ["inner","outer","center"]
			self.anchor_mode = modes.index(anchor)
		# anchor is int
		else:
			try:
				tfcomb.utils.check_value(anchor,0,2)
			except ValueError as e:
				self.logger.error(str(e))
				sys.exit(0)
			self.anchor_mode = anchor
		
	def count_distances(self, normalize = True, directional = False):
		""" Count distances for co_occurring TFs, can be followed by analyze_distances
			to determine preferred binding distances

		Parameters
		----------
		normalize : bool
			True if data should be normalized, False otherwise. Normalization is done as followed:
			(number of counted occurrences for a given pair at a given distance) / (Total amount of occurrences for the given pair)
			Default: True
		directional : bool
			Decide if direction of found pairs should be taken into account, e.g. whether  "<---TF1---> <---TF2--->" is only counted as 
			TF1-TF2 (directional=True) or also as TF2-TF1 (directional=False). Default: False.
		
		Returns
		----------
		None 
			Fills the object variable .distances.

		"""
		try:
			tfcomb.utils.check_type(normalize,[bool],"normalize")
		except ValueError as e:
			self.logger.error(str(e))
			sys.exit(0)

		try:
			tfcomb.utils.check_type(directional,[bool],"directional")
		except ValueError as e:
			self.logger.error(str(e))
			sys.exit(0)
		
		try:
			tfcomb.utils.check_type(self.min_dist,[int],"min_dist")
		except ValueError as e:
			self.logger.error(str(e))
			sys.exit(0)

		try:
			tfcomb.utils.check_type(self.max_dist,[int],"max_dist")
		except ValueError as e:
			self.logger.error(str(e))
			sys.exit(0)
		
		try:
			tfcomb.utils.check_type(self.anchor_mode,[int],"anchor_mode")
		except ValueError as e:
			self.logger.error(str(e))
			sys.exit(0)


		chromosomes = {site.chrom:"" for site in self.TFBS}.keys()
		# encode chromosome,pairs and name to int representation
		chrom_to_idx = {chrom: idx for idx, chrom in enumerate(chromosomes)}
		self.name_to_idx = {name: idx for idx, name in enumerate(self.TF_names)}
		sites = np.array([(chrom_to_idx[site.chrom], site.start, site.end, self.name_to_idx[site.name]) 
						  for site in self.TFBS]) #numpy integer array

		self.pairs_to_idx = {(self.name_to_idx[tf1],self.name_to_idx[tf2]): idx for idx, (tf1,tf2) in enumerate(self.rules[(["TF1","TF2"])].values.tolist())}
		
		self.logger.info("Calculating distances")
		self._raw = count_distances(sites, 
									self.pairs_to_idx,
									self.min_dist,
									self.max_dist,
									self.anchor_mode)
		
		# Unify (directional) counts 
		if not directional:
			for i in range(0,self._raw.shape[0]-1):
				if (self._raw[i,0] == self._raw[i+1,1]) and (self._raw[i,1] == self._raw[i+1,0]):
					s = self._raw[i,2:]+self._raw[i+1,2:]
					self._raw[i,2:] = s
					self._raw[i+1,2:] = s
		self.directional = directional

		# convert raw counts (numpy array with int encoded pair names) to better readable format (pandas DataFrame with TF names)
		self._raw_to_human_readable(normalize)

		self.logger.info("Done finding distances! Run .linregress_pair() or .linregress_all() to fit linear regression")
	
	def _raw_to_human_readable(self, normalize = True):
		""" Get the raw distance in human readable format
			
			Parameters
			----------
			normalize : bool
			True if data should be normalized, False otherwise. Normalization is done as followed:
			(number of counted occurrences for a given pair at a given distance) / (Total amount of occurrences for the given pair)
			Default: True

			Returns:
			----------
			pd.Dataframe (TF1 name, TF2 name, count min_dist, count min_dist +1, ...., count max_dist)
		"""
		self.logger.debug("Converting raw count data to pretty dataframe")
		idx_to_name = {}
		# get names from int encoding
		for k,v in self.name_to_idx.items():
			idx_to_name[v] = k 
		
		results = []
		for row in self._raw:
			tf1 = idx_to_name[row[0]]
			tf2 = idx_to_name[row[1]]
			entry = [tf1,tf2]
			
			if normalize:
				entry += (row[2:]/(row[2:].sum())).tolist()
			else:
				entry += row[2:].tolist()
			results.append(entry)

		self.normalized = normalize    
		self.distances = pd.DataFrame(results,columns=['TF1','TF2']+[str(x) for x in range (self.min_dist, self.max_dist+1)])

	def linregress_pair(self,pair,n_bins=None, save = None):
		""" Fits a linear Regression to distance count data for a given pair. The linear regression is used to 
			estimate the background. Proceed with .correct_pair()
			
			Parameters
			----------
			pair : tuple(str,str)
				TF names for which the linear regression should be performed. e.g. ("NFYA","NFYB")
			n_bins: int 
				Number of bins used for plotting. If n_bins is none, binning resolution is one bin per data point. 
				Default: None
			save: str
				Path to save the plots to. If save is None plots won't be plotted. 
				Default: None

			Returns:
			----------
			scipy.stats._stats_mstats_common.LinregressResult Object
		"""
		try:
			tfcomb.utils.check_type(pair,[tuple],"pair")
		except ValueError as e:
			self.logger.error(str(e))
			sys.exit(0)
		
		try:
			tfcomb.utils.check_type(n_bins,[int,type(None)],"n_bins")
		except ValueError as e:
			self.logger.error(str(e))
			sys.exit(0)

		if save is not None:
			try:
				tfcomb.utils.check_writeability(save)
			except ValueError as e:
				self.logger.error(str(e))
				sys.exit(0)

		if self.distances is None:
			self.logger.error("No distances evaluated yet. Please run .count_distances() first.")
			sys.exit(0)
		#TODO: check pair is valid
		tf1 = pair[0]
		tf2 = pair[1]

		self.logger.debug(f"Fitting linear regression for pair: {pair}")
		if n_bins is None:
			n_bins = self.max_dist - self.min_dist +1
		x = np.linspace(self.min_dist,self.max_dist+1, n_bins)
		
		data = self.distances.loc[((self.distances["TF1"]==tf1) &
			   (self.distances["TF2"]==tf2))].iloc[0, 2:]
		linres = linregress(range(self.min_dist,self.max_dist+1),np.array(data,dtype = float))
		if save is not None:
			plt.hist(range(self.min_dist,self.max_dist+1),weights=data, bins=n_bins, density=False, alpha=0.6)
			plt.plot(x, linres.intercept + linres.slope*x, 'r', label='fitted line')
			title = f"Fit results: pval = {linres.pvalue},  stderr = {linres.stderr}" 
			plt.title(title)
			plt.savefig(f'{save}linreg_{tf1}_{tf2}.png', dpi=600)
			plt.clf()
		return linres
	
	def linregress_all(self,n_bins = None, save = None):
		""" Fits a linear Regression to distance count data for all rules. The linear regression is used to 
			estimate the background. Proceed with .correct_all()
			
			Parameters
			----------
			n_bins: int 
				Number of bins used for plotting. If n_bins is none, binning resolution is one bin per data point. 
				Default: None
			save: str 
				Path to save the plots to. If save is None plots won't be plotted. 
				Default: None

			Returns:
			----------
			None
				Fills the object variable .linres
		"""

		try:
			tfcomb.utils.check_type(n_bins,[int,type(None)],"n_bins")
		except ValueError as e:
			self.logger.error(str(e))
			sys.exit(0)

		if save is not None:
			try:
				tfcomb.utils.check_writeability(save)
			except ValueError as e:
				self.logger.error(str(e))
				sys.exit(0)

		if self.distances is None:
			self.logger.error("No distances evaluated yet. Please run .count_distances() first.")
			sys.exit(0)
		self.logger.info("Fitting linear regression.")
		linres = {}
		for idx,row in self.distances.iterrows():
			tf1 = row["TF1"]
			tf2 = row["TF2"]
			res = self.linregress_pair((tf1,tf2),n_bins,save)
			linres[tf1,tf2]=[tf1,tf2,res]
		
		self.linres = pd.DataFrame.from_dict(linres,orient="index",columns=['TF1', 'TF2', 'Linear Regression']).reset_index(drop=True) 
	
	def correct_pair(self,pair,linres,n_bins = None, save = None):
		""" Subtracts the estimated background from the Signal for a given pair. 
			
			Parameters
			----------
			pair : tuple(str,str)
				TF names for which the background correction should be performed. e.g. ("NFYA","NFYB")
			linres: scipy.stats._stats_mstats_common.LinregressResult 
				Fitted linear regression for the given pair
			n_bins: int 
				Number of bins used for plotting. If n_bins is none, binning resolution is one bin per data point. 
				Default: None
			save: str
				Path to save the plots to. If save is None plots won't be plotted. 
				Default: None

			Returns:
			----------
			list 
				Corrected values for the given pair
		"""
		try:
			tfcomb.utils.check_type(pair,[tuple],"pair")
		except ValueError as e:
			self.logger.error(str(e))
			sys.exit(0)

		try:
			tfcomb.utils.check_type(n_bins,[int,type(None)],"n_bins")
		except ValueError as e:
			self.logger.error(str(e))
			sys.exit(0)

		if save is not None:
			try:
				tfcomb.utils.check_writeability(save)
			except ValueError as e:
				self.logger.error(str(e))
				sys.exit(0)

		#TODO: check pair is valid & check linres: import in utils needed ?
		tf1 = pair[0]
		tf2 = pair[1]

		if linres is None:
			self.logger.error("Please fit a linear regression first. [see .linregress_pair()]")
			sys.exit(0)

		self.logger.debug(f"Correcting background for pair {pair}")
		if n_bins is None:
			n_bins = self.max_dist - self.min_dist +1
	   
		if self.distances is None:
			self.logger.error("No distances evaluated yet. Please run .count_distances() first.")
			sys.exit(0)
		data = self.distances.loc[((self.distances["TF1"]==tf1) &
			   (self.distances["TF2"]==tf2))].iloc[0, 2:]
		corrected = []
		x_val = 0
		
		for dist in data:
			# subtract background from signal
			corrected.append(dist-(linres.intercept + linres.slope*x_val))
			x_val += 1

		if save is not None:
			x = np.linspace(self.min_dist,self.max_dist+1, n_bins)
			plt.hist(range(self.min_dist,self.max_dist+1),weights=corrected, bins=n_bins, density=False, alpha=0.6)
			linres = linregress(range(self.min_dist,self.max_dist+1),np.array(corrected,dtype = float))
			plt.plot(x, linres.intercept + linres.slope*x, 'r', label='fitted line')
			plt.savefig(f'{save}corrected_{tf1}_{tf2}.png', dpi=600)
			plt.clf()
		
		return corrected
	
	def correct_all(self,n_bins = None, save = None):
		""" Subtracts the estimated background from the Signal for all rules. 
			
			Parameters
			----------
			pair : tuple(str,str)
				TF names for which the background correction should be performed. e.g. ("NFYA","NFYB")

			Returns:
			----------
			None 
				Fills the object variable .corrected
		"""
		try:
			tfcomb.utils.check_type(n_bins,[int,type(None)],"n_bins")
		except ValueError as e:
			self.logger.error(str(e))
			sys.exit(0)

		if save is not None:
			try:
				tfcomb.utils.check_writeability(save)
			except ValueError as e:
				self.logger.error(str(e))
				sys.exit(0)
		
		if self.linres is None:
			self.logger.error("Please fit a linear regression first. [see .linregress_all()]")
			sys.exit(0)

		self.logger.info(f"Correcting background")
		corrected = {}
		
		for idx,row in self.linres.iterrows():
			tf1,tf2,linres = row
			res=self.correct_pair((tf1,tf2),linres,n_bins,save)
			corrected[tf1,tf2]=[tf1,tf2]+res
		
		self.corrected = pd.DataFrame.from_dict(corrected,orient="index",columns=['TF1','TF2']+[str(x) for x in range (self.min_dist, self.max_dist+1)]).reset_index(drop=True) 
		
	def get_median(self,pair):
		""" Estimates the median from the distinct counts per distance for a given pair.
			
			Parameters
			----------
			pair: tuple(str,str)
				TF names for which median should be calculated. e.g. ("NFYA","NFYB")
  
			Returns:
			----------
			Float 
				Median for the given pair 
		"""
		try:
			tfcomb.utils.check_type(pair,[tuple],"pair")
		except ValueError as e:
			self.logger.error(str(e))
			sys.exit(0)

		if self.distances is None:
			self.logger.error("Can not calculate Median, no distances evaluated yet. Please run .count_distances() first.")
			sys.exit(0)
		
		# TODO: check pair is valid
		tf1 = pair[0]
		tf2 = pair[1]

		data = self.distances.loc[((self.distances["TF1"]==tf1) &
			   (self.distances["TF2"]==tf2))].iloc[0, 2:]

		self.logger.debug(f" Median for pair {tf1} - {tf2}: {data.median}")
		return data.median()

	# TODO: Check if kwargs is better suited tham height & prominence
	def analyze_signal_pair(self, pair, corrected, smooth_window = 3, height = 0, prominence = 0, save = None, new_file = True):
		""" After background correction is done (see .correct_pair() or .correct_all()), the signal is analyzed for peaks, 
			indicating prefered binding distances. There can be more than one peak (more than one prefered binding distance) per 
			Signal. Peaks are called with scipy.signal.find_peaks().
			
			Parameters
			----------
			pair : tuple(str,str)
				TF names for which the background correction should be performed. e.g. ("NFYA","NFYB")
			corrected: list 
				corrected value for the given pair
			smooth_window: int 
				window size for the rolling smoothing window. A bigger window produces larger flanking ranks at the sides.
				(see tobias.utils.signals.fast_rolling_math) 
				Default: 3
			height: number or ndarray or sequence
				height parameter for peak calling (see scipy.signal.find_peaks() for detailed information). 
				Zero means only positive peaks are called.
				Default: 0
			prominence: number or ndarray or sequence
				prominence parameter for peak calling (see scipy.signal.find_peaks() for detailed information)
				Default: 0
			save: str
				Path to save the plots to. If save is None plots won't be plotted. 
				Default: None
			new_file: boolean
				True means results are written to a new file (overwrites already existing results), False means results are appended if 
				file already exists.
				Default: True

			Returns:
			----------
			list 
				list of found peaks in form [TF1, TF2, Distance, Peak Heights, Prominences, Prominence Threshold]
		"""
		try:
			tfcomb.utils.check_type(pair,[tuple],"pair")
		except ValueError as e:
			self.logger.error(str(e))
			sys.exit(0)

		try:
			tfcomb.utils.check_type(smooth_window,[int],"smooth_window")
		except ValueError as e:
			self.logger.error(str(e))
			sys.exit(0)
		
		try:
			tfcomb.utils.check_type(corrected,[list,pd.core.series.Series],"corrected")
		except ValueError as e:
			self.logger.error(str(e))
			sys.exit(0)

		if save is not None:
			try:
				tfcomb.utils.check_writeability(save)
			except ValueError as e:
				self.logger.error(str(e))
				sys.exit(0)

		tf1, tf2 = pair
		peaks = []
		if(smooth_window != 1):
			if smooth_window < 0 :
				self.logger.error("Window size need to be positive or zero.")
				sys.exit(0)
			smoothed = fast_rolling_math(np.array(list(corrected)), smooth_window, "mean")
			x = np.nan_to_num(smoothed)
		else:
			x = corrected
		
		peaks_smooth, properties = find_peaks(x, height=height, prominence = prominence)
		self.logger.debug(f"{len(peaks_smooth)} Peaks found")
		if (save is not None):
			if new_file:
				outfile = open(f'{save}peaks_{tf1}_{tf2}.tsv','w') 
				outfile.write(self._PEAK_HEADER)
			else:
				outfile = open(f'{save}peaks_{tf1}_{tf2}.tsv','a') 

		if (len(peaks_smooth) > 0):
			for i in range(len(peaks_smooth)):
				peak = [tf1,tf2,peaks_smooth[i],round(properties["peak_heights"][i],4),round(properties["prominences"][i],4),round(prominence,4)]
				peaks.append(peak)
				if (save is not None):
					outfile.write('\t'.join(str(x) for x in peak) + '\n')
		
		if (save is not None):
			outfile.close()

		return peaks
	
	def smooth(self,window_size = 3):
		""" Helper function for smoothing all rules with a given window size. The function .correct_all() is required to be run beforehand.
			
			Parameters
			----------
			window_size: int 
				window size for the rolling smoothing window. A bigger window produces larger flanking ranks at the sides.
				(see tobias.utils.signals.fast_rolling_math) 
				Default: 3

			Returns:
			----------
			None 
				Fills the object variable .smoothed
		"""
		try:
			tfcomb.utils.check_type(window_size,[int],"window size")
		except ValueError as e:
			self.logger.error(str(e))
			sys.exit(0)

		if window_size < 0 :
				self.logger.error("Window size need to be positive or zero.")
				sys.exit(0)
		
		if self.corrected is None:
			self.logger.error("Background is not yet corrected. Please try .correct_all() first.")
			sys.exit(0)
		all_smoothed = []
		
		self.smooth_window = window_size
		self.logger.info(f"Smoothing signal with window size {window_size}")
		for idx, row in self.corrected.iterrows():
			tf1 = row[0]
			tf2 = row[1]
			smoothed = fast_rolling_math(np.array(list(row[2:])), window_size, "mean")
			x = np.nan_to_num(smoothed)
			x = np.insert(np.array(x,dtype=object), 0, tf2)
			x = np.insert(x, 0, tf1)
			all_smoothed.append(x)
			
		self.smoothed = pd.DataFrame(all_smoothed,columns=['TF1','TF2']+[str(x) for x in range (len(all_smoothed[0])-2)])


	def analyze_signal_all(self, smooth_window = 3, height = 0, prominence = "median",save = None):
		""" After background correction is done (see .correct_all()), the signal is analyzed for peaks, 
			indicating prefered binding distances. There can be more than one peak (more than one prefered binding distance) per 
			Signal. Peaks are called with scipy.signal.find_peaks().
			
			Parameters
			----------
			smooth_window: int 
				window size for the rolling smoothing window. A bigger window produces larger flanking ranks at the sides.
				(see tobias.utils.signals.fast_rolling_math) 
				Default: 3
			height: number or ndarray or sequence
				height parameter for peak calling (see scipy.signal.find_peaks() for detailed information). 
				Zero means only positive peaks are called.
				Default: 0
			prominence: number or ndarray or sequence or "median"
				prominence parameter for peak calling (see scipy.signal.find_peaks() for detailed information). 
				If "median", the median for the pairs is used (see .get_median())
				Default: "median"
			save: str
				Path to save the plots to. If save is None plots won't be plotted. 
				Default: None
			new_file: boolean
				True means results are written to a new file (overwrites already existing results), False means results are appended if 
				file already exists.
				Default: True

			Returns:
			----------
			None 
				Fills the object variable self.peaks, self.smooth_window, self.peaking_count
		"""
		try:
			tfcomb.utils.check_type(smooth_window,[int],"smooth_window")
		except ValueError as e:
			self.logger.error(str(e))
			sys.exit(0)

		if save is not None:
			try:
				tfcomb.utils.check_writeability(save)
			except ValueError as e:
				self.logger.error(str(e))
				sys.exit(0)
		
		if smooth_window > 1:
			self.smooth(smooth_window)
		if self.corrected is None:
			self.logger.error("Background is not corrected yet. Please try .correct_all() first.")
			sys.exit(0)

		if isinstance(prominence,str):
			try:
				tfcomb.utils.check_string(prominence,["median"])
			except ValueError as e:
				self.logger.error(str(e))
				sys.exit(0)

		self.logger.info(f"Analyzing Signal")
		all_peaks = []

		if save is not None:
			outfile = open(f'{save}peaks.tsv','w')
			outfile.write(self._PEAK_HEADER)
		calc_mean = False
		if (prominence == "median"):
			calc_mean = True
	
		peaking_count = 0
		for idx,row in self.corrected.iterrows():
			tf1 = row["TF1"]
			tf2 = row["TF2"]
			corrected_data = self.corrected.loc[((self.corrected["TF1"]==tf1) &
												 (self.corrected["TF2"]==tf2))].iloc[0, 2:]
			
			if (calc_mean):
				prominence = self.get_median((tf1,tf2))

			peaks = self.analyze_signal_pair((tf1,tf2),
											  corrected_data, 
											  smooth_window = smooth_window, 
											  height = height, 
											  prominence = prominence, 
											  save = None)
				
			if len(peaks)>0:
				for peak in peaks:
					all_peaks.append(peak)
					if save is not None:    
						outfile.write('\t'.join(str(x) for x in peak) + '\n')
				peaking_count += 1

		self.peaks = pd.DataFrame(all_peaks,columns=self._PEAK_HEADER.strip().split("\t"))
		self.smooth_window = smooth_window
		self.peaking_count = peaking_count

		if save is not None:
			outfile.close()

	def is_smoothed(self):
		""" Return True if data was smoothed during analysis, False otherwise
			
			Parameters
			----------
		   
			Returns:
			----------
			bool 
				True if smoothed, False otherwiese
		"""
		
		if (self.smoothed is None) or (self.smooth_window <= 1): 
			return False
		return True
		

	def check_periodicity(self):
		""" checks periodicity of distances (like 10 bp indicating DNA full turn)
			- placeholder for functionality upgrade - 
		"""
		pass

	#-------------------------------------------------------------------------------------------------------------#
	#---------------------------------------------- plotting -----------------------------------------------------#
	#-------------------------------------------------------------------------------------------------------------#
	def plot_bar(self,targets,dataSource,save = None):
		""" Barplots for a list of TF-pairs

		 Parameters
			----------
			targets : array[tuple(str,str)]
				Pairs (tuples) to create plots for.
			dataSource : pd.DataFrame 
				Source Data (should be a result Table from .count_distances() e.g. .distances)
			save:
				Output path to write results to. (filename will be constructed automatically from TF1-/TF2-name)
				Default: None (results will not be saved)

		"""
		if save is not None:
			try:
				tfcomb.utils.check_writeability(save)
			except ValueError as e:
				self.logger.error(str(e))
				sys.exit(0)
		
		if not isinstance(dataSource,pd.DataFrame):
			try:
				source_table = pd.DataFrame(dataSource)
			except ValueError as e:
				self.logger.error(str(e))
				sys.exit(0)
			
		
		for pair in targets:
			#TODO: Check pair is valid
			fig = plt.bar(x = range(self.min_dist,self.max_dist+1), height=source_table.loc[((source_table["TF1"]==pair[0]) &
									   (source_table["TF2"]==pair[1]))].iloc[0, 2:])
			plt.title(pair)
			
			if save is not None:
				plt.savefig(f'{save}bar_{pair[0]}_{pair[1]}.png', dpi=600)
				plt.clf()

	def plot_hist(self,targets,dataSource,nbins=None,save = None):
		""" Histograms for a list of TF-pairs

		 Parameters
			----------
			targets : array[tuple(str,str)]
				Pairs (tuples) to create plots for.
			dataSource : pd.DataFrame 
				Source Data (should be a result Table from .count_distances(), e.g. .distances)
			nbins: int
				Number of bins. Default: None (Binning is done automatically)
			save:
				Output Path to write results to. (filename will be constructed automatically from TF1-/TF2-name)
				Default: None (results will not be saved)

		"""
		if save is not None:
			try:
				tfcomb.utils.check_writeability(save)
			except ValueError as e:
				self.logger.error(str(e))
				sys.exit(0)
		
		if not isinstance(dataSource,pd.DataFrame):
			try:
				source_table = pd.DataFrame(dataSource)
			except ValueError as e:
				self.logger.error(str(e))
				sys.exit(0)
			
		for pair in targets:
			# TODO: Check pair is valid
			plt.hist(range(self.min_dist,self.max_dist+1), nbins, weights=source_table.loc[((source_table["TF1"]==pair[0]) &
									   (source_table["TF2"]==pair[1]))].iloc[0, 2:])
			plt.title(pair)
			if save is not None:
				plt.savefig(f'{save}hist_{pair[0]}_{pair[1]}.png', dpi=600)
				plt.clf()

	def plot_dens(self,targets,dataSource,bwadjust = 1,save = None):
		""" KDE Plots for a list of TF-pairs

			Parameters
			----------
			targets : array[tuple(str,str)]
				Pairs (tuples) to create plots for.
			dataSource : pd.DataFrame 
				Source Data (should be a result Table from .count_distances(), e.g. .distances)
			bwadjust: float
				Factor that multiplicatively scales the value chosen using bw_method. Increasing will make the curve smoother. 
				See kdeplot() from seaborn. Default: 1
			save:
				Output Path to write results to. (filename will be constructed automatically from TF1-/TF2-name)
				Default: None (results will not be saved)

		"""
		try:
			tfcomb.utils.check_type(bwadjust,[tuple],"pair")
		except ValueError as e:
			self.logger.error(str(e))
			sys.exit(0)

		if save is not None:
			try:
				tfcomb.utils.check_writeability(save)
			except ValueError as e:
				self.logger.error(str(e))
				sys.exit(0)
		
		if not isinstance(dataSource,pd.DataFrame):
			try:
				source_table = pd.DataFrame(dataSource)
			except ValueError as e:
				self.logger.error(str(e))
				sys.exit(0)
		
		for pair in targets:
			weights = list(source_table.loc[((source_table["TF1"]==pair[0]) &
									   (source_table["TF2"]==pair[1]))].iloc[0, 2:])
			sns.kdeplot(range(self.min_dist,self.max_dist+1),weights = weights,bw_adjust=bwadjust,x="distance").set_title(pair)
			# TODO: Check if save is a valid path
			if save is not None:
				plt.savefig(f'{save}dens_{pair[0]}_{pair[1]}.png', dpi=600)
				plt.clf()

	def plot_analyzed_signal(self,pair, peaks = None, sourceData = None, save = None, only_peaking = False):
		if (sourceData is None) and (self.corrected is None):
			self.logger.error("Background is not yet corrected. Please try .correct_all() first or provide sourceData Table.")
			sys.exit(0)

		if (peaks is None) and (self.peaks is None):
			self.logger.error("Signal is not yet analyzed. Please try .analyze_signal_all() first or provide peak list.")
			sys.exit(0)

		tf1, tf2 = pair
		if peaks is None:
			peaks = self.peaks.loc[((self.peaks["TF1"]==tf1) &
									(self.peaks["TF2"]==tf2))].Distance.to_numpy()
		else:
			peaks = np.array(peaks)
		if sourceData is None: 
			if self.is_smoothed():
				x = self.smoothed.loc[((self.smoothed["TF1"]==tf1) &
									   (self.smoothed["TF2"]==tf2))].iloc[0,2:].to_numpy()
			else:    
				x = self.corrected.loc[((self.corrected["TF1"]==tf1) &
										(self.corrected["TF2"]==tf2))].iloc[0,2:].to_numpy()
		else:
			x = sourceData.loc[((sourceData["TF1"]==tf1) &
								(sourceData["TF2"]==tf2))].iloc[0,2:].to_numpy()
		if (only_peaking) and (len(peaks) == 0):
			return
			
		plt.plot (x)
		plt.plot(peaks, x[peaks], "x")
		plt.plot(np.zeros_like(x), "--", color="gray")
		plt.title(f"Analyzed signal for {tf1}-{tf2}")
		if save is not None:
			plt.savefig(f'{save}/peaks_{tf1}_{tf2}.png', dpi=600)
		plt.clf()
