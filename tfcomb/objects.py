

import os 
import pandas as pd
import itertools
import datetime
import multiprocessing as mp
import numpy as np
import collections
import copy
import glob
import fnmatch

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
import nxviz
import networkx as nx

#Utilities from TOBIAS
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
	The main class for collecting and working with co-occurring TFs
	"""

	def __init__(self, verbosity = 2): #set verbosity 

		#Function and run parameters
		self.verbosity = verbosity  #0: error, 1:warning, 2:info, 3:debug, 4:spam-debug
		self.logger = TFcombLogger(self.verbosity)
		
		#Variables for storing data
		self.prefix = None 	#is used when objects are added to a DiffCombObj
		self.TF_names = []		#List of TF names
		self.TF_counts = None 	#numpy array
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
		Internal method to add two CombObj together using: `CombObj1 + CombObj2 = new_CombObj`
		
		Parameters:
		----------
			obj : CombObj 
		
		Returns:
		----------
			A merged CombObj
		"""

		combined = CombObj(self.verbosity) #initialize empty combobj

		#Merge TFBS
		combined.TFBS = RegionList(self.TFBS + obj.TFBS)
		combined.TFBS.loc_sort() 				#sort TFBS by coordinates
		combined.TFBS = tfcomb.utils.remove_duplicates(combined.TFBS) #remove duplicated sites 

		return(combined)
	
	def copy(self):
		""" Returns a copy of the CombObj """

		copied = copy.copy(self)
		return(copied)
	
	def set_verbosity(self, level):
		""" Set the verbosity level for logging after creating the CombObj

		Parameters
		----------
		level : int
			A value between 0-4 where 0 (only errors), 1 (warning), 2 (info - default), 3 (debug), 4 (spam debug)
		"""

		self.verbosity = level
		self.logger = TFcombLogger(self.verbosity) #restart logger with new verbosity


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
		Function to calculate TFBS from motifs and genome fasta

		Parameters
		-----------
		regions : tobias.utils.regions.RegionList or str
			RegionList or str
		motifs : tobias.utils.motifs.MotifList or str 

		genome : str
			Path to the .fa genome to scan
		
		motif_pvalue : float
			Default: 0.0001
		
		motif_naming : str
			One of "name" (...)
		gc : float between 0-1

		keep_overlap : bool
			keep overlapping occurrences of the same TFBS,. Default: False

		threads : int
			For multiprocessing 


		Returns
		-----------
		None 
			Fills the .TFBS variable

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
		Fill the .TFBS attribute using a precalculated set of binding sites

		Parameters
		---------------
		bed_f : str 
			A path to a .bed-file with precalculated binding sites

		Returns
		---------------
		None
			the object attribute is set in place
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
		Fill the .TFBS variable with pre-calculated binding sites from TOBIAS BINDetect

		Parameters
		-----------
		bindetect_path : str
			Path to the BINDetect-output folder containing <TF1>, <TF2>, <TF3> (...) folders
		condition : str
			Name of condition to use

		Raises
		-------
		ValueError 

		Returns
		--------
		None 	

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
		

	def subset_TFBS(self, regions):
		"""
		Subset .TFBS in object to regions in bed

		Parameters
		-----------
		regions : str or RegionList


		Returns
		-------
		None - .TFBS is updated in place

		"""

		#
		if isinstance(regions, str):
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

	def count_within(self, window = 100, max_overlap = 0, stranded = False, directional = False):
		""" 
		Count co-occurrences between TFBS 

		current parameter set is seen using .print_parameters()

		Requir .TFBS to be filled by either `TFBS_from_motifs`, `TFBS_from_bed` or `TFBS_from_tobias`.
		
		Parameters
		-----------
		window : int
		
		max_overlap
			float between 0-1. Default: 0 (no overlap allowed)
		stranded
			Take strand of TFBSs into account
		directional : bool


		Returns
		----------
		None 
			Fills the object variable .counts


		See also
		----------
		market_basket()
		
		"""

		#Check that TFBS exist and that it is RegionList
		if self.TFBS is None or not isinstance(self.TFBS, RegionList):
			self.logger.error("No TFBS available in .TFBS . TFBS are set either using .precalculate_TFBS or .TFBS_from_bed")
			raise ValueError()
			
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
		self.logger.spam("TFBS names: {0}".format(self.TF_names))

		#Convert TFBS to internal numpy integer format
		chromosomes = {site.chrom:"" for site in TFBS}.keys()
		chrom_to_idx = {chrom: idx for idx, chrom in enumerate(chromosomes)}
		name_to_idx = {name: idx for idx, name in enumerate(self.TF_names)}
		sites = np.array([(chrom_to_idx[site.chrom], site.start, site.end, name_to_idx[site.name]) for site in TFBS]) #numpy integer array
	
		#Count number of bp covered by all TFBS
		self.TFBS_bp = len(self.TFBS) #get_unique_bp(self.TFBS)

		#---------- Count co-occurrences within TFBS ---------#

		self.logger.debug("Counting co-occurrences within sites")
		TF_counts, pair_counts = count_co_occurrence(sites, window, max_overlap, n_TFs)
		pair_counts = tfcomb.utils.make_symmetric(pair_counts) if directional == False else pair_counts	#Deal with directionality

		self.TF_counts = TF_counts
		self.pair_counts = pair_counts

		self.rules = None #Remove .rules if market_basket() was previously run
		self.logger.info("Done finding co-occurrences! Run .market_basket() to estimate significant pairs")


	def count_between(self, bedpe_file, directional=False):
		"""
		Function to count co-occurring TFs between 

		Note
		-----
		Requires the '.TFBS' to be filled e.g. using self.TFBS_from_bed

		Parameters
		-----------
		bedpe_file : str
			Path to a .bedpe-formatted file containing region1-region2 interactions

		directional : bool
			Whether the direction of the pair TF1-TF2 / TF2-TF1 matters. If directional == True, TF1 will always be in the first 
			bedpe-region, and TF2 will be in the second.

		"""
		
		#Check that TFBS exist and that it is RegionList
		if self.TFBS is None or not isinstance(self.TFBS, RegionList):
			self.logger.error("No TFBS available in .TFBS . TFBS are set either using .precalculate_TFBS or .TFBS_from_bed")
			return
			
		self.logger.info("Counting co-occurring TFs from .TFBS...")

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


	def get_pair_locations(self, TF1, TF2, window = 100, max_overlap = 0):
		""" Get genomic locations of a particular motif pair. Requires .TFBS to be filled.
		
		Parameters
		----------
		TF1 : str 
			Name of TF1 in pair
		TF2 : str 
			Name of TF2 in pair
		window : int

		Returns
		-------
		RegionList()
			All locations of TF1-TF2 (range is TF1(start)->TF2(end)

		"""

		### Check that .TFBS is filled

		locations = RegionList() #empty regionlist

		TFs = (TF1, TF2)
		w = window
		sites = self.TFBS
		n_sites = len(sites)

		#Loop over all sites
		i = 0
		while i < n_sites: #i is 0-based index, so when i == n_sites, there are no more sites
			
			#Get current TF information
			TF1_chr, TF1_start, TF1_end, TF1_name = sites[i].chrom, sites[i].start, sites[i].end, sites[i].name

			if TF1_name not in TFs:
				i += 1
				continue #next i_site
	
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
					TF2_chr, TF2_start, TF2_end, TF2_name = sites[i+j].chrom, sites[i+j].start, sites[i+j].end, sites[i+j].name
					
					if TF2_name not in TFs:
						j += 1
						continue #next j in finding_assoc

					#True if these TFBS co-occur within window
					if TF1_chr == TF2_chr and (TF2_end - TF1_start <= w):
						
						# check if they are overlapping more than the threshold
						valid_pair = 1
						overlap_bp = TF1_end - TF2_start
						
						# Get the length of the shorter TF
						short_bp = min([TF1_end - TF1_start, TF2_end - TF2_start])
						
						#Invalid pair, overlap is higher than threshold
						if overlap_bp / short_bp > max_overlap: 
							valid_pair = 0

						#Save association
						if valid_pair == 1:
							location = OneRegion([TF1_chr, TF1_start, TF2_end, TF1_name + "_" + TF2_name])
							locations.append(location)

					else:
						#The next site is out of window range; increment to next i
						i += 1
						finding_assoc = False   #break out of finding_assoc-loop
		
		return(locations)

	#-----------------------------------------------------------------------------------------#
	#-------------------------------- Market basket analysis ---------------------------------#
	#-----------------------------------------------------------------------------------------#

	def market_basket(self, sort_by="cosine", threads = 1):
		"""
		Runs market basket analysis on the counts within .counts()

		Requires prior run of .count_between() or .count_within()
	

		Parameters
		-----------
		sort_by : str
		
		threads : int


		"""

		#Check that TF counts are available
		if (self.TF_counts is None) or (self.pair_counts is None):
			raise ValueError()


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
		table["jaccard"] = table["TF1_TF2_support"] / (table["TF1_support"] + table["TF2_support"] - table["TF1_TF2_support"])
		
		#Remove rows with TF1_TF2_count == 0
		table = table[table["TF1_TF2_count"] != 0]

		#Sort for highest measure pairs
		table.sort_values(sort_by, ascending=False, inplace=True)
		table.reset_index(inplace=True, drop=True)

		#Market basket is done; save to .rules
		self.logger.info("Market basket analysis is done! Results are found in .rules")
		self.rules = table

	def calculate_pvalues(self, measure="cosine"):
		""" """

		#Assign pvalue
		self.logger.debug("Calculating p-value for {0} rules".format(len(self.rules)))
		self.rules[measure + "_pvalue"] = tfcomb.utils._calculate_pvalue(self.rules, measure=measure)


	def select_rules(self, measure="cosine", 
							pvalue="cosine_pvalue", 
							measure_threshold=None,
							pvalue_threshold=0.05,
							plot=True):

		"""
		Make selection of rules based on distribution of measure and pvalue.

		Parameters
		-----------
		measure : str, optional
			Default: 'cosine'
		pvalue : str, optional
			Column 'cosine_pvalue'

		plot : bool, optional
			Whether to show the plot or not

		Returns
		--------
		Pandas.DataFrame()
			A subset of <obj>.rules

		See also
		---------
		tfcomb.objects.plot_background
		tfcomb.plotting.volcano
		"""

		#Check if measure / log2fc are in columns
		columns = [measure, pvalue]
		for col in columns:
			if col not in self.rules.columns:
				raise KeyError("'{0}' is not in .rules".format(col))

		#If measure_threshold is None; try to calculate optimal threshold via knee-plot
		if measure_threshold == None:
			
			#Compute distribution histogram of measure values
			y, x = np.histogram(self.rules[measure], bins=1000)
			x = [np.mean([x[i], x[i+1]]) for i in range(len(x)-1)] #Get mid of start/end of each bin
			y = np.cumsum(y)
			kneedle = KneeLocator(x, y, curve="concave", direction="increasing")
			measure_threshold = kneedle.knee

		#Set threshold on table
		selected = self.rules.copy()
		selected = selected[(selected[measure] >= measure_threshold) & (selected[pvalue] <= pvalue_threshold)]

		if plot == True:
			tfcomb.plotting.volcano(self.rules, measure = measure, 
												pvalue = pvalue, 
												measure_threshold = measure_threshold,
												pvalue_threshold = pvalue_threshold)

		return(selected)


	#-----------------------------------------------------------------------------------------#
	#-------------------------------- Plotting functionality  --------------------------------#
	#-----------------------------------------------------------------------------------------#

	def plot_background(self, pair, measure="cosine"):
		"""
		plot background values for TF pair

		Parameters
		----------
		pair : tuple
			Tuple of (TF1, TF2)

		measure : str


		"""


		background = ""

		_ = plt.hist(background, bins=20, density=True, label="Background pairs containing either {0} or {1}")
		plt.plot(xvals, p, 'k', linewidth=2)
		plt.axvline(obs, color="red", label="({0},{1})")
		plt.xlabel("log(lift)")
		plt.title((TF1, TF2))
		plt.show()


	def plot_heatmap(self, n_rules=20, color_by="cosine", sort_by=None, figsize=(8,8)):
		"""
		Wrapper for the plotting function `tfcomb.plotting.heatmap`

		Parameters
		-----------
		n_rules : int
			The number of rules to show. The first `n_rules` rules of .rules are taken. Default: 20
		color_by : str
			A column within .rules to color the heatmap by. Note: Can be different than sort_by. Default: "cosine"	

		sort_by : str, optional
			A column within .rules to sort by before choosing n_rules
		

		See also
		---------
		tfcomb.plotting.heatmap
		"""

		#Check types

		#Check that columns are available in self.rules
		columns = [col for col in [color_by, sort_by] if col is not None]
		for col in columns:
			if col not in self.rules.columns:
				raise ValueError()
				
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
		tfcomb.plotting.heatmap(chosen_associations, color_by=color_by, figsize=figsize)


	def plot_bubble(self, n_rules=20, sort_by="cosine", yaxis="cosine", color_by="confidence", size_by="TF1_TF2_support", figsize=(8,8)):
		"""
		Wrapper for the plotting function `tfcomb.plotting.bubble`
		
		Parameters
		-----------
		n_rules : int

		See also
		-----------
		tfcomb.plotting.bubble
		"""

		#Sort rules
		sorted_table = self.rules.sort_values(sort_by, ascending=False)

		#Select n top rules
		top_rules = sorted_table.head(n_rules)
		top_rules.index = top_rules["TF1"].values + " + " + top_rules["TF2"].values

		#Plot
		ax = tfcomb.plotting.bubble(top_rules, yaxis=yaxis, color_by=color_by, size_by=size_by, figsize=figsize)

		return(ax)
	

	#-------------------------------------------------------------------------------------------#
	#------------------------------------ Network analysis -------------------------------------#	
	#-------------------------------------------------------------------------------------------#

	def plot_network(self, n_rules=20, color_edge_by="cosine", color_node_by="TF1_count", layout="spring_layout"): 
		"""
		Plot the rules in .rules as a network

		Parameters
		-----------
		n_rules : int

		layout : str

		See also
		--------
		tfcomb.analysis.build_network and tfcomb.plotting.network
		"""

		#Create subset of rules
		selected = self.rules[:n_rules]

		#Build network
		G = tfcomb.analysis.build_network(selected)
		
		#Plot network
		tfcomb.plotting.network(G, layout=layout, color_edge_by=color_edge_by, color_node_by=color_node_by)


	def plot_circos(self, n_rules=20, color_edge_by="lift", size_edge_by="lift", color_node_by=None, size_node_by=None):
		"""

		Parameters
		-----------
		n_rules 

		Note
		-----------
		Requires run of market_masket()

		"""
		
		#Choose subset of nodes
		subset = self.rules[:n_rules]

		#Create network
		G = tfcomb.analysis.build_network(subset)

		#Plot circos 
		tfcomb.plotting.circos(G) #, node1="TF1", node2="TF2")

		#subset = self.rules[["TF1", "TF2", size_edge_by]][:n_rules]

		#Setup network
		#self._build_network(n_rules = n_rules)
		#select edge with highest weight
		#selected_edges = [tuple(pair) for pair in self.rules[["TF1", "TF2"]][:n_rules].to_numpy()]
		#sub = self.nx.edge_subgraph(selected_edges)

		#Plot circos
		#c = nxviz.CircosPlot(self.nx, node_labels=True, fontfamily="sans-serif") #, edge_width=size_edge_by)  #, node_color='affiliation', node_grouping='affiliation')
		#c.draw()	

	#.group_TFs (based on network)
	#although it is only pairs, hubs can be found of TF-cocktails. 

	#-----------------------------------------------------------------------------------------#
	#------------------------------ Comparison to other objects ------------------------------#
	#-----------------------------------------------------------------------------------------#

	def compare(self, obj_to_compare, normalize=True):
		"""
		Utility function to create a DiffCombObj directly

		Requires market basket run on both objects

		Parameters
		---------
		obj_to_compare : tfcomb.objects.CombObj

		normalize : bool
			Whether to normalize 
	
		"""
		
		#TODO: Check that market basket was run on both objects

		
		diff = DiffCombObj([self, obj_to_compare])

		if normalize == True:
			diff.normalize("cosine")

		diff.calculate_foldchanges(normalize=normalize) #also calculates p-values

		return(diff)



###################################################################################
############################## Differential analysis ##############################
###################################################################################


class DiffCombObj():

	def __init__(self, objects = []):
		""" Initializes a DiffCombObj object for doing differential analysis between CombObjs.

		Parameters
    	----------
		objects : list
			A list of CombObj instances

		See also
		---------
		add_object for adding objects one-by-one

		"""
		
		#Initialize object variables
		self.n_objects = 0
		self.prefixes = [] #filled by ".add_object"
		self.measures_to_keep = ["confidence", "lift", "cosine"] #list of measures to keep when adding together objects

		#Setup logger
		self.verbosity = 3
		self.logger = TFcombLogger(self.verbosity)

		#Add objects one-by-one
		for obj in objects:
			self.add_object(obj)


	def __str__(self):
		pass

	def add_object(self, obj):
		"""
		Add one CombObj to the DiffCombObj 

		Parameters
		-----------
		obj : CombObj
			An instance of CombObj
		"""

		#TODO: Check that object is an instance of CombObj

		#Check if prefix is set - otherwise, set to obj<int>
		if obj.prefix is not None:
			prefix = obj.prefix
		else:
			prefix = "Obj" + str(self.n_objects + 1)
			#logger warning

		#Format table from obj to contain TF1/TF2 + measures with prefix
		columns_to_keep = ["TF1", "TF2"] + self.measures_to_keep
		obj_table = obj.rules[columns_to_keep] #only keep necessary columns
		obj_table.rename(columns={col: prefix + "_" + col for col in self.measures_to_keep}, inplace=True)

		#Initialize table if this is the first object
		if self.n_objects == 0: 
			self.rules = obj_table

		#Or add object to this DiffCombObj
		else:
			self.rules = self.rules.merge(obj_table, left_on=["TF1", "TF2"], right_on=["TF1", "TF2"], how="outer")
			self.rules = self.rules.fillna(0) #Fill NA with null (happens if TF1/TF2 pairs are different between objects)

		self.n_objects += 1 #current number of objects +1 for the one just added
		self.prefixes.append(prefix)
		

	def normalize(self, measure):
		"""
		Normalize the values for the given measure using quantile normalization

		Parameters
		-----------
		measure : str

		Returns 
		-------

		"""
		
		#TODO: check available measures
		#available measures = 

		#Establish input/output columns
		measure_columns = [prefix + "_" + measure for prefix in self.prefixes]
		measure_columns_norm = [col + "_norm" for col in measure_columns]

		#Normalize values
		self.rules[measure_columns_norm] = qnorm.quantile_normalize(self.rules[measure_columns], axis=1)


	def calculate_foldchanges(self, measure="cosine", normalize=False, pseudo = 1):
		""" Calculate foldchanges between objects in DiffCombObj 
		
		Parameters
		----------
		measure : str, optional
			Column from .rules to calculate foldchange on

		normalized : bool
			Whether to use across-object normalized values for the calculation. Default False.

		pseudo : float
			The pseudocount to add to all values before log2-foldchange transformation. Default: 1.

		See also
		--------
		tfcomb.DiffCombObj.normalize
		"""

		table = self.rules

		#Find all possible combinations of objects
		combinations = itertools.combinations(self.prefixes, 2)
		
		columns = [] #collect the log2fc columns per contrast
		for p1, p2 in combinations:
			log2_col = "{0}_{1}_{2}_log2fc".format(p1, p2, measure)
			columns.append(log2_col)

			if normalize:
				p1_values = self.rules[p1 + "_" + measure + "_norm"]
				p2_values = self.rules[p2 + "_" + measure + "_norm"]
			else:
				p1_values = self.rules[p1 + "_" + measure]
				p2_values = self.rules[p2 + "_" + measure]

			pseudo = 1

			self.rules[log2_col] = np.log2((p1_values + pseudo) / (p2_values + pseudo))

			#Calculate p-value of each pair
			pvalue_col = "{0}_{1}_{2}_pvalue".format(p1, p2, measure)
			self.rules[pvalue_col] = tfcomb.utils._calculate_pvalue(self.rules, measure=log2_col, alternative="two-sided")

		#Sort by first contrast log2fc
		self.rules.sort_values(columns[0], inplace=True)
		

	def select_rules(self, measure, plot = True):
		"""
		Select differentially regulated rules on the basis of measure and pvalue
		"""




		if plot == True:
			tfcomb.plotting.volcano(self.rules, measure=measure)
			
	

	#-------------------------------------------------------------------------------------------#
	#----------------------------- Plots for differential analysis -----------------------------#
	#-------------------------------------------------------------------------------------------#

	def plot_correlation(self, measure="cosine", method="pearson"):
		"""
		Plot correlation between rules across objects for the measure given.

		Parameters
		-----------
		measure : str
			Default: "cosine" 
		method : str
			pearson or spearman

		See also 
		----------
		tfcomb.objects.DiffCombObj.correlate
		"""

		#Define columns
		cols = [prefix + "_" + measure for prefix in self.prefixes]

		#Calculate matrix and plot
		matrix = self.rules[cols].corr(method=method)
		sns.clustermap(matrix,
							cbar_kws={'label': method})



	def plot_heatmap(self, n_rules=10, sort_by=None, color_by=None):
		"""
		Functionality to plot a heatmap of differentially co-occurring TF pairs. 

		Parameters
		------------
		n_rules : int
			Number of rules to show from each contrast (default: 10). Note: This is the number of rules either up/down, meaning that the rules shown are n_rules * 2.

		See also
		----------
		tfcomb.plotting.heatmap
		"""

		#Check if columns are found in table
		cols = [col for col in [sort_by, color_by] if col is not None]
		for col in cols:
			if col not in self.rules.columns:
				raise ValueError("Column '{0}' is not found in .rules".format(col))

		#Establish color_by
		if color_by is None:
			pass
			#
			#color_by = [col for col in self.rules.columns if measure + "_log2fc" in col]

		#Sort by
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


	def plot_bubble(self, n_rules=20, sort_by="confidence_log2fc", yaxis="confidence_log2fc",
																   color_by="lift_log2fc", 
																   size_by="lift_log2fc", figsize=(7,7)):
		"""
		Plot bubble scatterplot 

		See also
		----------
		tfcomb.plotting.bubble
		"""

		#Select top/bottom n rules
		sorted_table = self.rules.sort_values(sort_by, ascending=False)
		sorted_table.index = sorted_table["TF1"] + " + " + sorted_table["TF2"]

		top_rules = sorted_table.head(n_rules)
		bottom_rules = sorted_table.tail(n_rules)

		# Draw each cell as a scatter point with varying size and color
		fig, (ax1, ax2) = plt.subplots(2, figsize=figsize, constrained_layout=True) 

		tfcomb.plotting.bubble(data=top_rules, yaxis=yaxis, color_by=color_by, size_by=size_by, ax=ax1)
		tfcomb.plotting.bubble(data=bottom_rules, yaxis=yaxis, color_by=color_by, size_by=size_by, ax=ax2)

		#Remove x-axis label for upper plot

		return(fig)

	def plot_network(self):
		pass

