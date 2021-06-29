

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

#Statistics
import qnorm #quantile normalization
import scipy
from scipy.stats import rankdata
from scipy.stats import norm
import statsmodels.stats.multitest

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

#TF-comb modules
import tfcomb
import tfcomb.plotting
import tfcomb.analysis
from tfcomb.counting import count_co_occurrence
from tfcomb.logging import *
from tfcomb.utils import *

from kneed import KneeLocator

class CombObj(): 
	"""
	The main class for collecting and working with co-occurring TFs.

	Examples
    ----------

	>>> C = tfcomb.objects.CombObj()

	#Verbosity of the output log can be set using the 'verbosity' parameter:
	>>> C = tfcomb.objects.CombObj(verbosity=2)

	"""

	def __init__(self, verbosity = 1): #set verbosity 

		#Function and run parameters
		self.verbosity = verbosity  #0: error, 1:info, 2:debug, 3:spam-debug
		self.logger = TFcombLogger(self.verbosity)
		
		#Variables for storing data
		self.prefix = None 	#is used when objects are added to a DiffCombObj
		self.TF_names = []		#List of TF names
		self.TF_counts = None 	#numpy array of size n_TFs
		self.pair_counts = None	#numpy matrix of size n_TFs x n_TFs
		self.n_bp = 0			#predict the number of baskets 
		self.TFBS = RegionList() #None RegionList() of TFBS
		self.rules = None  		#filled in by .market_basket()

		#Formatted data / open files for reading
		self._genome_obj = None
		self._motifs_obj = None

	def __str__(self):

		s = "Instance of CombObj:\n"
		#s += "- genome: {0}\n".format(self.genome)
		#s += "- motifs: {0}\n".format(self.motifs, )

		s += "- TFBS: {0}".format(len(self.TFBS))
		#s += "- Market basket analysis: {0}\n".format(self.mb_analysis) 
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

		filehandler = open(path, 'r') 
		obj = pickle.load(filehandler)

		#Check if object is CombObj
		if not isinstance(obj, CombObj):
			raise TypeError("Read object from '{0}' is not a CombObj".format(path))

		#Overwrite self with CombObj
		self = obj

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
			How to name TFs based on input motifs. Must be one of "name", ". Default: "name".
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
		self.TFBS = RegionList()	#initialize empty list
		n_regions = len(regions)
		
		self.logger.info("Scanning for TFBS with {0} core(s)...".format(threads))

		#Define whether to run in multiprocessing or not
		if threads == 1:

			chunks = regions.chunks(100) 
			genome_obj = tfcomb.utils.open_genome(genome)	#open pysam fasta obj

			n_regions_processed = 0
			for region_chunk in chunks:
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
			
			#TODO: Print progress
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
		---------------
		bed_f : str 
			A path to a .bed-file with precalculated binding sites.

		"""

		self.logger.info("Reading sites from '{0}'...".format(bed_f))
		self.TFBS = RegionList().from_bed(bed_f)
		n_sites = len(self.TFBS)

		#Stats on the regions which were read
		counts = {r.name: "" for r in self.TFBS}
		n_names = len(counts)

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
			self.TFBS += RegionList().from_bed(f)

		#Process TFBS
		self.TFBS.loc_sort()

		self.logger.info("Read {0} sites from condition '{1}'".format(len(self.TFBS), condition))
		

	def cluster_TFBS(self, threshold=0.5):
		""" Cluster TFBS based on overlap 
		
		Parameters
		------------
		threshold : float
		
		"""

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

		self.logger.info("")


	def subset_TFBS(self, regions, bedpe=False):
		"""
		Subset .TFBS in object to specific regions. Can be used to select only a subset of TFBS (e.g. only in promoters) to run analysis on.

		Parameters
		-----------
		regions : str or RegionList
			Path to a .bed-file containing regions or a tobias-format RegionList object. 
		bedpe : bool
			Whether or not the given .bed-file is a .bedpe file. If 'True', TFBS will be subset by both paired regions. If False (default), 
			.bed-file will be assumed to contain coordinates for only one region per line (columns 1-3).

		Returns
		-------
		None
			The .TFBS attribute is updated in place

		"""

		#If regions are string, read to internal format
		if isinstance(regions, str):
			#todo: bedpe
			regions = RegionList().from_bed(regions)
		
		#Create regions->sites dict
		TFBS_in_regions = assign_sites_to_regions(self.TFBS, regions)

		#Merge across keys
		self.TFBS = RegionList(sum([TFBS_in_regions[key] for key in TFBS_in_regions], []))
		self.TFBS.loc_sort()


	def TFBS_to_bed(self, path):
		"""
		Writes out the .TFBS regions to a file. This is a wrapper for the tobias.utils.regions.RegionList().write_bed() utility.

		Parameters
		----------
		path : str
			File path to write .bed-file to.
		"""	

		#TODO: Check if bed is writeable
		self.TFBS.write_bed(path)


	#------------------------------------------------------------------------------------------------------------#
	#----------------------------------------- Counting co-occurrences -------------------------------------------#
	#-------------------------------------------------------------------------------------------------------------#

	def count_within(self, min_distance = 0, 
						   max_distance = 100, 
						   max_overlap = 0, 
						   stranded = False, 
						   directional = False, 
						   binarize = False):
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

		Returns
		----------
		None 
			Fills the object variables .TF_counts and .pair_counts.
		
		Raises
		--------
		ValueError
			If .TFBS has not been filled.
		"""

		#Check that TFBS exist and that it is RegionList
		if self.TFBS is None or not isinstance(self.TFBS, RegionList):
			raise ValueError("No TFBS available in '.TFBS'. The TFBS are set either using .TFBS_from_motifs, .TFBS_from_bed or TFBS_from_TOBIAS")
			
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

		self.rules = None 	#Remove .rules if market_basket() was previously run
		self.logger.info("Done finding co-occurrences! Run .market_basket() to estimate significant pairs")


	def count_between(self, bedpe_file, directional=False):
		"""
		Function to count co-occurring TFs between paired regions, e.g. promoter-enhancer interactions. This function requires .TFBS to be filled by either 
		`TFBS_from_motifs`, `TFBS_from_bed` or `TFBS_from_tobias`. This function can be followed by .market_basket to calculate association rules.

		Parameters
		-----------
		bedpe_file : str
			Path to a .bedpe-formatted file containing region1-region2 interactions.
		directional : bool, optional
			Whether the direction of the pair TF1-TF2 / TF2-TF1 matters. If directional == True, TF1 will always be in the first 
			.bedpe-region, and TF2 will be in the second. Default: False.
		"""
		
		#Check that TFBS exist and that it is RegionList
		if self.TFBS is None or not isinstance(self.TFBS, RegionList):
			raise ValueError("No TFBS available in .TFBS . TFBS are set either using .TFBS_from_motifs, .TFBS_from_bed or TFBS_from_TOBIAS")

		#Read bedpe format
		bedpe = pd.read_csv(bedpe_file, sep="\t", header=None)
		from_regions = bedpe.iloc[:,0:3]
		to_regions = bedpe.iloc[:,3:6]
		self.logger.info("Read {0} interactions from .bedpe file".format(len(bedpe)))

		#Convert to RegionList format
		from_regions_obj = RegionList([OneRegion(row.values) for idx, row in from_regions.iterrows()])
		to_regions_obj = RegionList([OneRegion(row.values) for idx, row in to_regions.iterrows()])

		all_regions_obj = RegionList(from_regions_obj + to_regions_obj)
		all_regions_obj = all_regions_obj.remove_duplicates() #remove duplicates arising from multiple interactions to same regions
		all_regions_obj.write_bed("all_regions.bed")

		#Assign TFBS to each region
		TFBS_in_regions = tfcomb.utils.assign_sites_to_regions(self.TFBS, all_regions_obj)
		self.logger.info("Assigned .TFBS to {0} regions".format(len(TFBS_in_regions)))

		#Create lists of interactions e.g. [(gene1_promoter, gene1_enhancers), (gene2_promoter, gene2_enhancers), (...)]
		interactions = zip(from_regions_obj, to_regions_obj)
		
		self.logger.info("Counting interactions between {0} pairs of regions".format(bedpe.shape[0]))

		TF_dict = {}
		pair_dict = {}
		for (region1, region2) in interactions:
			r1_tup = (region1.chrom, region1.start, region1.end)
			r2_tup = (region2.chrom, region2.start, region2.end)

			r1_TFBS = TFBS_in_regions.get(r1_tup, None)
			r2_TFBS = TFBS_in_regions.get(r2_tup, None)
			
			if r1_TFBS is None or r2_TFBS is None:
				continue # cannot count trans-co-occurrence when sites are missing

			r1_TFs = list({TFBS.name: 0 for TFBS in r1_TFBS}.keys())
			r2_TFs = list({TFBS.name: 0 for TFBS in r2_TFBS}.keys())

			#Save single TFs:
			for TF in r1_TFs + r2_TFs:
				TF_dict[TF] = TF_dict.get(TF, 0) + 1

			#Fetch all combinations of sites
			for pair in itertools.product(r1_TFs, r2_TFs):
				pair_dict[pair] = pair_dict.get(pair, 0) + 1
		
		#Convert dicts to numpy arrays
		self.logger.debug("Converting TF/pair counts to internal format")
		self.TF_names = list(TF_dict.keys())
		n_TFs = len(self.TF_names)
		self.logger.debug("TF_names = {0}".format(self.TF_names))
		self.TF_counts = np.array([TF_dict[TF] for TF in self.TF_names])

		#Initialize dataframe with 0's and fill from pair_dict
		pair_df = pd.DataFrame(np.zeros((n_TFs, n_TFs)), 
											columns = self.TF_names, 
											index = self.TF_names)
		for (TF1, TF2) in pair_dict:
			pair_df.loc[TF1, TF2] = pair_dict[(TF1, TF2)]

		self.pair_counts = pair_df.to_numpy()
		self.pair_counts = tfcomb.utils.make_symmetric(self.pair_counts) if directional == False else self.pair_counts	#Deal with directionality
		
		#self.n_regions = bedpe.shape[0] #number of interactions
		self.TFBS_bp = len(self.TFBS) 
		self.rules = None 				#market basket must be rerun if it was previously run

		self.logger.info("Finished .count_between()! Run .market_basket() to estimate significant pairs")


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
		RegionList()
			All locations of TF1-TF2 (range is TF1(start)->TF2(end))

		See also
		---------
		count_within

		"""

		### TODO: Check that .TFBS is filled

		locations = RegionList() #empty regionlist

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
										location = OneRegion([TF1_chr, TF1_start, TF2_end, TF1_name + "_" + TF2_name])
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

	def market_basket(self, sort_by="cosine", threads = 1):
		"""
		Runs market basket analysis on the TF1-TF2 counts. Requires prior run of .count_within() or .count_between().
	
		Parameters
		-----------
		sort_by : str, optional
			Measure to sort the final rules by. Default: 'cosine'.
		threads : int, optional
			Threads to use for multiprocessing. Default: 1.

		Raises
		-------
		ValueError 
			If no TF counts are available.
		"""

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

		#Calculate confidence / lift / additional metrics
		table["confidence"] = table["TF1_TF2_support"] / table["TF1_support"]
		table["lift"] = table["confidence"] / table["TF2_support"]
		table["cosine"] = table["TF1_TF2_support"] / np.sqrt(table["TF1_support"] * table["TF2_support"])
		#table["jaccard"] = table["TF1_TF2_support"] / (table["TF1_support"] + table["TF2_support"] - table["TF1_TF2_support"])
		
		#Remove rows with TF1_TF2_count == 0
		table = table[table["TF1_TF2_count"] != 0]

		#Sort for highest measure pairs
		table.sort_values(sort_by, ascending=False, inplace=True)
		table.reset_index(inplace=True, drop=True)

		#Market basket is done; save to .rules
		self.logger.info("Market basket analysis is done! Results are found in .rules")
		self.rules = table

	def calculate_pvalues(self, measure="cosine"):
		""" Calculate pvalues for all """

		#Assign pvalue
		self.logger.debug("Calculating p-value for {0} rules".format(len(self.rules)))
		self.rules[measure + "_pvalue"] = tfcomb.utils._calculate_pvalue(self.rules, measure=measure)

	def select_rules(self, measure="cosine", 
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
			Default: 'cosine'
		pvalue : str, optional
			Column 'cosine_pvalue'
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
		Pandas.DataFrame()
			A subset of <obj>.rules

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
		n_rules : int
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

		#Check that columns are available in self.rules
		check_columns(self.rules, [color_by, sort_by])
				
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


	def plot_bubble(self, n_rules=20, yaxis="cosine", color_by="confidence", size_by="TF1_TF2_support", sort_by=None, 
					unique = True, **kwargs):
		"""
		Plot a bubble-style scatterplot of the object rules. This is a wrapper for the plotting function `tfcomb.plotting.bubble`.
		
		Parameters
		-----------
		n_rules : int, optional
			The number of rules to show. The first `n_rules` rules of .rules are taken. Default: 20.
		yaxis : str, optional
			A column within .rules to depict on the y-axis of the plot. Default: "cosine".	
		color_by : str, optional
			A column within .rules to color points in the plot by. Default: "confidence".
		size_by : str, optional
			A column within .rules to size points in the plot by. Default: "TF1_TF2_support".
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

		#Whether to keep TF1-TF2 and TF2-TF1 pairs (or whether to only show unique)
		if unique == True:
			table.set_index(["TF1", "TF2"], inplace=True)
			pairs = table.index

			#Collect unique pairs (first occurrence is kept)
			to_keep = {}
			for pair in pairs:
				if not pair[::-1] in to_keep: #if opposite was not already found
					to_keep[pair] = ""

			#Subset table
			table = table.loc[list(to_keep.keys())]
			table.reset_index(inplace=True)

		#Select n top rules
		top_rules = table.head(n_rules)
		top_rules.index = top_rules["TF1"].values + " + " + top_rules["TF2"].values

		#Plot
		ax = tfcomb.plotting.bubble(top_rules, yaxis=yaxis, color_by=color_by, size_by=size_by, **kwargs)

	

	#-------------------------------------------------------------------------------------------#
	#------------------------------------ Network analysis -------------------------------------#	
	#-------------------------------------------------------------------------------------------#

	def plot_network(self, n_rules=100, 
						   color_node_by="TF1_count",
						   color_edge_by="cosine", 
						   size_edge_by="TF1_TF2_count",
						   **kwargs): 
		"""
		Plot the rules in .rules as a network using Graphviz for python. This function is a wrapper for 
		building the network (using tfcomb.analysis.build_network) and subsequently plotting the network (using tfcomb.plotting.network).

		Parameters
		-----------
		n_rules : int, optional
			The number of rules to show within network (selected as the top 'n_rules' within .rules). Default: 100.
		color_node_by : str, optional
			A column in .rules to color nodes by. Default: 'TF1_count'.
		color_edge_by : str, optional
			A column in .rules to color edges by. Default: 'cosine'.
		size_edge_by : str
			A column in rules to size edge width by. Default: 'TF1_TF2_count'.
		kwargs : arguments
			All other arguments are passed to tfcomb.plotting.network.

		See also
		--------
		tfcomb.analysis.build_network and tfcomb.plotting.network
		"""

		#Create subset of rules
		selected = self.rules[:n_rules]

		#Build network
		G = tfcomb.analysis.build_nx_network(selected)
		
		#Plot network
		dot = tfcomb.plotting.network(G, color_node_by=color_node_by, color_edge_by=color_edge_by, size_edge_by=size_edge_by, 
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

		#TODO: check that prefixes are unique; otherwise, throw error 

		#Check if prefix is set - otherwise, set to obj<int>
		if obj.prefix is not None:
			prefix = obj.prefix
		else:
			prefix = "Obj" + str(self.n_objects + 1)
			#logger warning

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

		self.n_objects += 1 #current number of objects +1 for the one just added
		self.prefixes.append(prefix)
		

	def normalize(self):
		"""
		Normalize the values for the given measure using quantile normalization. 
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
		

	def select_rules(self, contrast=None,
						   measure="cosine", 
						   plot = True):
		"""
		Select differentially regulated rules on the basis of measure and pvalue.

		Parameters
		-----------
		contrast : tuple
			Name of the contrast to use in tuple format e.g. (<prefix1>,<prefix2>). Default: None (the first contrast is shown).

		measure : str
			The measure to use for selecting rules. Default: "cosine".

		See also
		----------
		tfcomb.plotting.volcano
		"""


		if plot == True:
			tfcomb.plotting.volcano(self.rules, measure=measure)
			
	

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
			Default: "cosine" (converted to "prefix1/prefix2_<color_by>")
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
							n_rules=100, 
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
		n_rules : 100, optional
			The number of rules to show in either direction. Default: 100.
		color_node_by : str, optional
			Name of measure to color node by. If column is not in .rules, the name will be internally converted to "prefix1/prefix2_<color_edge_by>". Default: None.
		size_node_by : str, optional
			Column in .rules to size_node_by. 
		color_edge_by : str, optional
			The name of measure or column to color edge by. Default: "cosine_log2fc" (will be internally converted to "prefix1/prefix2_<color_edge_by>")
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
		selected = pd.concat([self.rules[:n_rules], self.rules[-n_rules:]])

		#Build network
		self.logger.debug("Building network using 'tfcomb.analysis.build_network'")
		G = tfcomb.analysis.build_nx_network(selected)
		
		#Plot network
		self.logger.debug("Plotting network using 'tfcomb.plotting.network'")
		dot = tfcomb.plotting.network(G, color_node_by=color_node_by, size_node_by=size_node_by, 
										 color_edge_by=color_edge_by, size_edge_by=size_edge_by, 
										 verbosity = self.verbosity, **kwargs)

		return(dot)

