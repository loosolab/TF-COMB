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
from tfcomb.counting import count_co_occurrence
from tfcomb.logging import *
from tfcomb.utils import *



class CombObj(): 

	"""
	The main class for collecting and working with co-occurring TFs

    Attributes
    ----------


	"""

	def __init__(self):

		#Parameters for TFBS
		self.keep_overlapping = False
		self.stranded = False
		self.directional = False 

		#Parameters for the background for counting
		self.n_background = 100
		self.background_pair_mean = None #updated during run if n_background > 0
		self.background_pair_std = None  #updated during run if n_background > 0

		#Paths for motif scanning
		self.genome = None
		self.motifs = None
		self.motif_pvalue = 0.0001
		self.motif_naming = "name"
		self.gc = 0.5

		#Formatted data / open files for reading
		self.genome_obj = None
		self.motifs_obj = None
		
		#Co-occurrence data obtained
		self.TF_names = []
		self.TF_counts = {}
		self.pair_counts = {}
		self.n_bp = 0
		self.TFBS = RegionList() #RegionList() of TFBS

		#Function and run parameters
		self.cores = 1
		self.verbosity = 2  #0: error, 1:warning, 2:info, 3:debug, 4:spam-debug
		self.logger = TFcombLogger(self.verbosity)

		#Flags for completed analysis
		self.mb_analysis = False
		self.measure = "cosine"
		self.table = None

		#Set default function parameters
		#self.add_counts = True
		self.keep_overlaps = False
		self.window = 100
		self.max_overlap = 0 #float between 0-1
		self.stranded = False
		self.directional = False
		#self #simplify #simplify stranded/directional analysis to scenarios #facing, #convergent, TF1_TF2, TF2-TF1
		
		#self.binarize = False

		#self.parallel = False  #is this run in parallel?
		#self.save_TFBS = False 
		#self.inplace = True 		

		#Count of time
		self.timing = {task: datetime.timedelta(0) for task in ["fetch", "scan", "collecting", "co-occurrence", "overlaps"]}

	#def __getstate__():

	def __str__(self):

		s = "Instance of CombObj:\n"
		#s += "- genome: {0}\n".format(self.genome)
		#s += "- motifs: {0}\n".format(self.motifs, )


		s += "- TFBS: {0}".format(len(self.TFBS))
		#s += "- Regions counted: {0} ({1} bp)\n".format(self.n_regions, self.n_bp)
		#s += "- Market basket analysis: {0}\n".format(self.mb_analysis) 
		return(s)

	def __repr__(self):
		return(self.__str__())
	

	def __add__(self, obj):
		"""
		Internal method to add two CombObj together using: `CombObj1 + CombObj2 = new_CombObj`
		
		Parameters:
		----------
		obj (CombObj): 
		
		Returns:
		----------
		ComObj
		"""

		combined = self.copy() #initialize empty combobj

		#Merge TFBS
		combined.TFBS = self.TFBS + obj.TFBS
		combined.TFBS.loc_sort() #sort
		#remove overlapping sites

		#Check whether TF names are the same in both ; decides how counts should be merged
		obj1_TFs = set(self.TF_names)
		obj2_TFs = set(obj.TF_names)

		"""
		self = CombObj() #initialize empty combobj
		for obj in list_of_objs:
			self.n_regions += obj.n_regions
			self.n_bp += obj.n_bp
			self.add_count_dict(obj.counts)
			self.timing = merge_dicts([self.timing, obj.timing])

			for region_id in obj.TFBS:
				self.TFBS[region_id] = self.TFBS.get(region_id, RegionList([])) + obj.TFBS[region_id]

		return(self)
		"""
		#common, setojb1_TFs
		#extend with those missing
		
		#common + obj1 specific + obj2 specific
		
		return(combined)
	
	def copy(self):
		""" Returns a copy of the CombObj """

		copied = copy.copy(self)
		return(copied)
	
	def set_verbosity(self, level):
		""" Set the verbosity level for logging

		Parameters
		----------
			level : int
				A value between 0-4 where 0 (only errors), (...)
		"""
		self.verbosity = level
		self.logger = TFcombLogger(self.verbosity) #restart logger with new verbosity

	def print_parameters(self):
		"""
		Print the current parameter set for this CombObj
		"""

		s = "" #initialize output string
		#.name = 

		#Module parameters
		#s += "- add_counts: {0}\n".format(self.add_counts)
		s += "- window: {0}\n".format(self.window)
		s += "- max_overlap: {0}\n".format(self.max_overlap)
		s += "- stranded: {0}\n".format(self.stranded)
		s += "- directional: {0}\n".format(self.directional)
		#s += "- binarize: {0}\n".format(self.binarize)
		s += "- motif_pvalue: {0}\n".format(self.motif_pvalue)
		s += "- motif_naming: {0}\n".format(self.motif_naming)

		s += "- verbosity: {0} (set using '.set_verbosity(<level>)')\n".format(self.verbosity)

		#full explanation of parameters is found at (...)
		print(s)


	#-------------------------------------------------------------------------------------------#
	#-------------------------------- Preparing for functions ----------------------------------#
	#-------------------------------------------------------------------------------------------#

	def validate_parameters(self):
		""" Validates whether the given parameters are valid, e.g. if cores = integer etc. 

			Raises valueerror if parameters are not valid.
		"""
		#Check if files are existing


	def _open_genome(self):	
		""" """

		#Initalize 
		self.genome_obj = pysam.FastaFile(self.genome)

	def _close_genome(self):
		""" """
		
		if self.genome_obj is not None:
			self.genome_obj.close()
			self.genome_obj = None

	def _prepare_motifs(self):
		""" """

		#Read and prepare motifs
		self.motifs_obj = MotifList().from_file(self.motifs)
		self.logger.debug("Read {0} motifs from '{1}'".format(len(self.motifs_obj), self.motifs))

		_ = [motif.get_threshold(self.motif_pvalue) for motif in self.motifs_obj]
		_ = [motif.set_prefix(self.motif_naming) for motif in self.motifs_obj] #using naming from args


	#-------------------------------------------------------------------------------#
	#-------------------------- Setting up the .TFBS list --------------------------#
	#-------------------------------------------------------------------------------#

	def TFBS_from_motifs(self, regions):
		"""
		Function to calculate TFBS from motifs and genome fasta

		Parameters
		-----------
		regions : :obj:str  
			RegionList or str
		
		Returns
		-----------
		None 
			.precalculate_TFBS fills the .TFBS variable

		"""

		s = datetime.datetime.now()

		#check that genome and motifs are set
		if self.motifs == None:
			raise ValueError(".motifs not set")
		if self.genome == None:
			raise ValueError(".genome not set")
	

		if isinstance(regions, str):
			regions_f = regions
			regions = RegionList().from_bed(regions)
			self.logger.debug("Read {0} regions from {1}".format(len(regions), regions_f))

		self._prepare_motifs()

		self.TFBS = RegionList()	#initialize empty list
		n_regions = len(regions)
		
		chunks = regions.chunks(self.cores * 2) #creates chunks of regions for multiprocessing
		self.logger.info("Scanning for TFBS with {0} core(s)...".format(self.cores))

		#Define whether to run in multiprocessing or not
		if self.cores == 1:
			self._open_genome()	#open pysam fasta obj

			n_regions_processed = 0
			for region_chunk in chunks:
				for region in regions:
					region_TFBS = self._calculate_TFBS(regions, close_genome=False)
					self.TFBS += region_TFBS

				#Update progress
				n_regions_processed += len(region_chunk)
				self.logger.debug("{0:.2f}% ({1} / {2})".format((n_regions_processed/n_regions*100, n_regions_processed, n_regions)))

			self._close_genome()

		else:

			#Setup pool
			pool = mp.Pool(self.cores)
			jobs = []
			for chunk in chunks:
				jobs.append(pool.apply_async(self._calculate_TFBS, (chunk,)))
			pool.close()
			
			#Print progress
			results = [job.get() for job in jobs]

			#Join all TFBS to one list
			self.TFBS = RegionList(sum(results, []))
			#self.TFBS = RegionList

		#Process TFBS
		self._process_TFBS()

		self.logger.info("Identified {0} TFBS within given regions".format(len(self.TFBS)))
		e = datetime.datetime.now()


	def _calculate_TFBS(self, regions, close_genome=True):
		"""
		Multiprocessing-safe function run from "TFBS_from_motifs"

		Parameters
		----------
		regions : RegionList()
			A RegionList() object of regions 

		Returns
		----------
		RegionList of TFBS within regions

		"""

		self._open_genome() #open the genome given
		TFBS = RegionList()

		for region in regions:
			seq = self.genome_obj.fetch(region.chrom, region.start, region.end)
			region_TFBS = self.motifs_obj.scan_sequence(seq, region)
			region_TFBS.loc_sort()

			TFBS += region_TFBS
		
		if close_genome == True:
			self._close_genome()
		
		return(TFBS)


	def TFBS_from_bed(self, bed_f):
		"""
		Fill the .TFBS attribute using a precalculated set of binding sites

		Parameters
		---------------
		bed_f : str 
			A path to a .bed-file with precalculated binding sites

		Returns
		---------------
		None - the object attribute is set in place
		"""

		self.logger.info("Reading sites from '{0}'...".format(bed_f))
		self.TFBS = RegionList().from_bed(bed_f)
		n_sites = len(self.TFBS)

		#Stats on the regions which were read
		counts = {r.name: "" for r in self.TFBS}
		n_names = len(counts)

		#Process TFBS
		self._process_TFBS()

		self.logger.info("Read {0} sites (comprising {1} unique TFs)".format(n_sites, n_names))


	def TFBS_from_TOBIAS(self, bindetect_path, condition):
		"""

		Parameters
		-----------
		bindetect_path : str
			Path to the BINDetect-output folder containing <TF1>, <TF2>, <TF3> (...) folders
		condition : str
			Name of condition to use

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
		self._process_TFBS()

		self.logger.info("Read {0} sites from condition '{1}'".format(len(self.TFBS), condition))
		

	def subset_TFBS(self, regions):
		"""
		Subset .TFBS in object to regions in bed

		Parameters
		-----------

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


	def _process_TFBS(self):
		"""
		Function to process the TFBS found in .TFBS
		"""

		
		#Remove overlapping for the same factor?
		if self.keep_overlaps == False:
		
			"""
			#Collect TFBS per TF
			s = datetime.datetime.now()
			sites_per_TF = {}
			for site in TFBS:
				name = site.name
				sites_per_TF[name] = sites_per_TF.get(name, []) + [site]
			self.timing["collecting"] += datetime.datetime.now() - s

			#Remove overlapping TFBS for the same factors (e.g. with +/- palindromic motifs)
			s = datetime.datetime.now()
			resolved = RegionList()
			for TF in sites_per_TF:
				resolved.extend(RegionList(sites_per_TF[TF]).resolve_overlaps())
			self.timing["overlaps"] += datetime.datetime.now() - s
			resolved.loc_sort()
			"""


	def TFBS_to_bed(self, path):
		"""
		Wrapper for the RegionList().write_bed() utility

		Parameters
		----------
		path : str
			File path to write .bed-file to

		Returns
		--------
		None
		"""	
		self.TFBS.write_bed(path)


	#-------------------------------------------------------------------------------------------------------------#
	#----------------------------------------- Counting co-occurrences -------------------------------------------#
	#-------------------------------------------------------------------------------------------------------------#

	def count_within(self):
		""" 
		Count co-occurrences between TFBS 

		current parameter set is seen using .print_parameters()

		Note
		------
		Requires .TFBS to be filled by either 
		
		
		Returns
		----------
		None 
			Fills the object variable .counts


		See also
		----------
		market_basket()
		
		"""

		self.validate_parameters() 	#Self.cores must be integer

		#Check that TFBS exist and that it is RegionList
		if self.TFBS is None or not isinstance(self.TFBS, RegionList):
			self.logger.error("No TFBS available in .TFBS . TFBS are set either using .precalculate_TFBS or .TFBS_from_bed")
			return
			
		self.logger.info("Counting co-occurring TFs from .TFBS...")

		#Should strand be taken into account?
		TFBS = copy.copy(self.TFBS)
		if self.stranded == True:
			for site in TFBS:
				site.name = "{0}({1})".format(site.name, site.strand)

		#Find all names within TFBS 
		self.TF_names = sorted(list(set([site.name for site in TFBS]))) #ensures that the same TF order is used across cores/subsets
		n_TFs = len(self.TF_names)

		#Convert TFBS to internal numpy integer format
		chromosomes = {site.chrom:"" for site in TFBS}.keys()
		chrom_to_idx = {chrom: idx for idx, chrom in enumerate(chromosomes)}
		name_to_idx = {name: idx for idx, name in enumerate(self.TF_names)}
		sites = np.array([(chrom_to_idx[site.chrom], site.start, site.end, name_to_idx[site.name]) for site in TFBS]) #numpy integer array
	
		#Count number of bp covered by all TFBS
		self.TFBS_bp = len(self.TFBS) #get_unique_bp(self.TFBS)

		#---------- Count co-occurrences within TFBS ---------#
		self.logger.debug("Counting co-occurrences within sites")
		s = datetime.datetime.now()
		TF_counts, pair_counts = count_co_occurrence(sites, self.window, self.max_overlap, n_TFs)
		pair_counts = tfcomb.utils.make_symmetric(pair_counts) if self.directional == False else pair_counts	#Deal with directionality

		self.TF_counts = TF_counts
		self.pair_counts = pair_counts
		self.timing["co-occurrence"] += datetime.datetime.now() - s

		#---------- Count background co-occurrences ------#
		if self.n_background > 0:

			#Find out with how many cores to run
			self.logger.debug("Running with cores = {0}".format(self.cores))

			pool = mp.Pool(self.cores)

			jobs = []
			np.random.seed(2021)
			for i in range(self.n_background):
				
				#Shuffle names in place
				np.random.shuffle(sites[:,3])
				
				#Find co-occurrence for background
				self.logger.debug("Applying background co-occurrence job for i={0}".format(i))
				jobs.append(pool.apply_async(count_co_occurrence, (sites, self.window, self.max_overlap, n_TFs)))
			
			#Update progress

			#Wait for background to finish
			pool.close() #no more jobs added
			pool.join()	#waits for jobs to finish

			#Collect background counts
			background_pair_counts = np.zeros((n_TFs, n_TFs, len(jobs))) #three dimensional stack of background counts
			for i, job in enumerate(jobs):
				_, pair_counts = job.get() #only pair counts are needed; TF_counts stay the same
				pair_counts = tfcomb.utils.make_symmetric(pair_counts) if self.directional == False else pair_counts	#Deal with directionality
				background_pair_counts[:,:,i] = pair_counts

			#Calculate mean and std of background
			self.background_pair_mean = np.mean(background_pair_counts, axis=2)
			self.background_pair_std = np.std(background_pair_counts, axis=2)

		self.table = None
		self.logger.info("Done finding co-occurrences! Run .market_basket() to estimate significant pairs")


	def count_between(self, bedpe_file):
		"""
		Function to count co-occurring TFs between 

		Note
		-----
		Requires the '.TFBS' to be filled e.g. using self.TFBS_from_bed

		Parameters
		-----------
		bedpe_file : str
			Path to a .bedpe-formatted file containing region1-region2 interactions

		"""

		self.validate_parameters()
		
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
		all_regions_obj.remove_duplicates() #remove duplicates

		#Assign TFBS to each region
		TFBS_in_regions = tfcomb.utils.assign_sites_to_regions(self.TFBS, all_regions_obj)
		self.logger.info("Assigned .TFBS to {0} regions".format(len(TFBS_in_regions)))

		#Create lists of tuples e.g. [(gene1_promoter, gene1_enhancers), (gene2_promoter, gene2_enhancers), (...)]
		tuples = zip(from_regions_obj, to_regions_obj)
		
		self.logger.info("Counting interactions between {0} pairs of regions".format(bedpe.shape[0]))

		TF_dict = {}
		pair_dict = {}
		for (region1, region2) in tuples:
			r1_tup = (region1.chrom, str(region1.start), str(region1.end))
			r2_tup = (region2.chrom, str(region2.start), str(region2.end))

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
		self.TF_counts = np.array([TF_dict[TF] for TF in self.TF_names])

		pair_df = pd.DataFrame(np.zeros((n_TFs, n_TFs)), 
											columns = self.TF_names, 
											index = self.TF_names)
		print(pair_df)

		for (TF1, TF2) in pair_dict:
			pair_df.loc[TF1, TF2] = pair_dict[(TF1, TF2)]
		
		print(pair_df)

		self.pair_counts = pair_df.to_numpy()
		self.pair_counts = self._make_symmetric(self.pair_counts) if self.directional == False else self.pair_counts	#Deal with directionality
		
		#self.n_regions = bedpe.shape[0] #number of interactions
		self.TFBS_bp = len(self.TFBS) 
		self.table = None 				#market basket must be rerun if it was previously run

		return(self)

	def get_pair_locations(self, TF1, TF2):
		""" Get genomic locations of a particular motif pair 
		
		Parameters
		----------
		TF1 : str 
			Name of TF1 in pair
		TF2 : str 
			Name of TF2 in pair
		
		Returns
		-------
		RegionList()
			All locations of TF1-TF2 (range is TF1(start)->TF2(end)

		"""

		locations = RegionList() #empty regionlist

		TFs = (TF1, TF2)
		w = self.window
		max_overlap = self.max_overlap
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

	def market_basket(self):
		"""
		Runs market basket analysis on the counts within .counts()

		Note
		------
		Requires prior run of .count_between() or .count_within()


		"""
		#if count_dict == None:
		#	print(self.count_dict)

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

		#Calculate joined co-occurence score
		#n_rules = table.shape[0]
		#table["lift_percentile"] = rankdata(table["lift"], method="min") / n_rules
		#table["confidence_percentile"] = rankdata(table["confidence"], method="min") / n_rules
		#table["co_occurrence_score"] = table[["confidence_percentile", "lift_percentile"]].mean(axis=1)
		#table.drop(columns=["confidence_percentile", "lift_percentile"], inplace=True)

		#Calculate p-value of pair
		if self.n_background is not None and self.n_background > 0:
			
			self.background_pair_std[self.background_pair_std == 0] = np.nan
			z = (self.pair_counts - self.background_pair_mean) / self.background_pair_std
			z[np.isnan(z)] = 0 #background_pair_std == 0
			z[z < 0] = 0	#only significant if pair counts > background
			p = scipy.stats.norm.sf(np.abs(z)) * 2
			pvalue_table = pd.DataFrame(p, index=self.TF_names, columns=self.TF_names)
			pvalue_table["TF1"] = pvalue_table.index
			pvalue_melted = pd.melt(pvalue_table, id_vars=["TF1"], var_name=["TF2"], value_name="pvalue")

			#Add pvalue to table
			table = table.merge(pvalue_melted, left_on=["TF1", "TF2"], right_on=["TF1", "TF2"])

			#Correct pvalues
			table["pvalue_adj"] = statsmodels.stats.multitest.multipletests(table["pvalue"], method="Bonferroni")[1]

		else:
			table[["pvalue", "pvalue_adj"]] = np.nan #pvalues not calculated

		#Remove rows with TF1_TF2_count == 0
		table = table[table["TF1_TF2_count"] != 0]

		#Sort for highest measure pairs
		table.sort_values(self.measure, ascending=False, inplace=True)
		table.reset_index(inplace=True, drop=True)

		self.table = table
		self.mb_analysis = True
	

	def make_selection(self, measure=None, plot=False):
		"""
		Make selection based on distribution of measure

		Parameters
		-----------

		Returns
		--------
		Pandas.DataFrame()
			a subset of self.table
		"""

		#Check if table exists
		if measure == None:
			measure = self.measure


		data = np.array(sorted(self.table[measure]))
		x = np.arange(len(data))

		#Find knee
		kneedle = KneeLocator(x, data, S=1.0, curve="convex", direction="increasing")

		#Set threshold on table
		threshold = kneedle.knee
		selected = self.table.copy()
		selected = selected[selected[measure] >= threshold]

		if plot == True:
			plt.axvline(kneedle.knee, color="red")
			plt.scatter(x, data)
			plt.ylabel(measure)
			plt.xlabel("rank")

		return(selected)

	
	def compare_to_background(self, pair, plot=True):
		"""
		Note
		-------
		Requires the .table attribute (from .market_basket() run)

		Parameters
		----------
		pair : tuple
			Tuple of (TF1, TF2)

		plot : bool
			Default True

		Returns
		--------
		pvalue

		"""








		return(pvalue)




	#-----------------------------------------------------------------------------------------#
	#------------------------------ Comparison to other objects ------------------------------#
	#-----------------------------------------------------------------------------------------#

	def compare(self, obj_to_compare):
		"""
		Parameters
		---------
		obj_to_compare : tfcomb.objects.CombObj

		Note
		------
		Requires market basket run on both objects
		"""
		
		#Check that market basket was run on both objects
		if self.mb_analysis != True:
			pass
		if obj_to_compare.mb_analysis != True:
			pass

		
		diff = DiffCombObj([self, obj_to_compare])
		#diff.normalize()
		diff._foldchange()

		return(diff)


	#-----------------------------------------------------------------------------------------#
	#-------------------------------- Plotting functionality  --------------------------------#
	#-----------------------------------------------------------------------------------------#

	def plot_heatmap(self, n_rules=20, sort_by="lift", color_by="lift", figsize=(8,8)):
		"""
		Wrapper for the plotting function tfcomb.plotting.heatmap
		"""
	
		#Check that columns are available in self.table
		if sort_by not in self.table.columns:
			raise ValueError()
		
		#Sort table
		associations = self.table.sort_values(sort_by, ascending=False)

		#Choose n number of rules
		tf1_list = associations["TF1"][:n_rules]
		tf2_list = associations["TF2"][:n_rules]

		# Fill all combinations for the TFs selected from top rules (to fill white spaces)
		chosen_associations = associations[(associations["TF1"].isin(tf1_list) &
								associations["TF2"].isin(tf2_list))]

		#Plot
		tfcomb.plotting.heatmap(chosen_associations, color_by=color_by)


	def plot_bubble(self, n_rules=20, sort_by="confidence", yaxis="confidence", color_by="lift", size_by="TF1_TF2_support", figsize=(8,8)):
		"""
		"""

		#Sort rules
		sorted_table = self.table.sort_values(sort_by, ascending=False)

		#Select n top rules
		top_rules = sorted_table.head(n_rules)
		#top_rules["label"] = top_rules["TF1"].values + " + " + top_rules["TF2"].values
		top_rules.index = top_rules["TF1"].values + " + " + top_rules["TF2"].values #.set_index("label", inplace=True)

		#Plot
		ax = tfcomb.plotting.bubble(top_rules, yaxis=yaxis, color_by=color_by, size_by=size_by, figsize=figsize)

		return(ax)
	
	def plot_scatter(self, xaxis="", yaxis=""):

		pass


	#-------------------------------------------------------------------------------------------#
	#------------------------------------ Network analysis -------------------------------------#	
	#-------------------------------------------------------------------------------------------#



		self.nx = MG

	def plot_network(self, layout="spring_layout"): 
		"""

		"""


		#Build network
		tfcomb.analysis.build_network(self.table)
		

		available_networks = []
		pass
		#drawing.layout.spring_layout)

	def plot_circos(self, n_rules=20, color_edge_by="lift", size_edge_by="lift", color_node_by=None, size_node_by=None):
		"""

		Parameters:
		-----------
		n_rules 

		Note
		-----------
		Requires run of market_masket()

		"""
		
		#Choose subset of nodes
		subset = self.table[:n_rules]

		tfcomb.plotting.circos(subset, node1="TF1", node2="TF2")

		#subset = self.table[["TF1", "TF2", size_edge_by]][:n_rules]

		#Setup network
		#self._build_network(n_rules = n_rules)
		#select edge with highest weight
		#selected_edges = [tuple(pair) for pair in self.table[["TF1", "TF2"]][:n_rules].to_numpy()]
		#sub = self.nx.edge_subgraph(selected_edges)

		#Plot circos
		#c = nxviz.CircosPlot(self.nx, node_labels=True, fontfamily="sans-serif") #, edge_width=size_edge_by)  #, node_color='affiliation', node_grouping='affiliation')
		#c.draw()	

	#.group_TFs (based on network)
	#although it is only pairs, hubs can be found of TF-cocktails. 


###################################################################################
############################## Differential analysis ##############################
###################################################################################


class DiffCombObj():

	def __init__(self, objects = [], prefixes = None):
		""" Initializes a DiffCombObj object for doing differential analysis between CombObjs.

		See also: Add_object for adding objects one-by-one

		Parameters
    	----------
		objects : list
			A list of CombObj instances

		prefixes : list
			A list of same length as objects for naming 
		"""
		
		self.n_objects = 0
		self.prefixes = [] #filled by ".add_object"
		
		#Setup logger
		self.verbosity = 3
		self.logger = TFcombLogger(self.verbosity)

		#Check that length of prefixes is the same as objects
		if prefixes is not None:
			if len(prefixed) != len(objects):
				raise ValueError("")
		else:
			prefixes = ["Obj" + str(n) for n in range(1, len(objects)+1)]	
		#self.prefixes = prefixes

		#Add obj one-by-one
		for i, obj in enumerate(objects):
			self.add_object(obj, prefixes[i])

		#
		self.measure = "cosine"
		self.measures = ["confidence", "lift", "cosine"]
		self.normalized = False


	def add_object(self, obj, prefix = None):
		"""
		
		"""

		#Check if prefix is set - otherwise, set to obj<int>
		if prefix is None:
			prefix = "Obj" + str(self.n_objects + 1)

		columns_to_keep = ["TF1", "TF2", "confidence", "lift", "cosine"]

		#Format table 
		obj_table =  obj.table.copy()
		obj_table = obj_table[columns_to_keep] #only keep necessary columns
		obj_table.columns = ["TF1", "TF2"] + [prefix + "_" + col for col in obj_table.columns if col not in ["TF1", "TF2"]]

		#Initialize table
		if self.n_objects == 0: 
			self.table = obj_table

		#Or add object to thisDiffCombObj
		else:
			self.table = self.table.merge(obj_table, left_on=["TF1", "TF2"], right_on=["TF1", "TF2"], how="outer")

		self.n_objects += 1
		self.prefixes.append(prefix)

		#Fill NA with null
		self.table = self.table.fillna(0)

		#Normalize between all conditions
		#self.normalize()
		

	def plot_correlation(self):
		""" 
		Parameters
    	----------

		Examples
        --------

		"""

	def normalize(self):
		"""
		"""

		#All lift and confidence columns
		lift_columns = [prefix + "_lift" for prefix in self.prefixes]
		confidence_columns = [prefix + "_confidence" for prefix in self.prefixes]

		#Normalize lift and confidence columns
		lift_columns_norm = [col + "_norm" for col in lift_columns]
		self.table[lift_columns_norm] = qnorm.quantile_normalize(self.table[lift_columns], axis=1)

		confidence_columns_norm = [col + "_norm" for col in confidence_columns]
		self.table[confidence_columns_norm] = qnorm.quantile_normalize(self.table[confidence_columns], axis=1)
		
		#Table has been normalized
		self.normalized = True

		return(self)


	def _foldchange(self):
		""" compare between two objects """

		table = self.table
		
		for measure in self.measures:
			columns = [prefix + "_" + measure for prefix in self.prefixes]
			self.logger.debug("{0} columns: {1}".format(measure, columns))

			if self.normalized:
				columns = [col + "_norm" for col in columns]

			#Calcualate log2fc
			c1, c2 = columns
			self.table[measure + "_log2fc"] = np.log2((table[c1] + 0.01) / (table[c2] + 0.01))

		return(self)


	#-------------------------------------------------------------------------------------------#
	#----------------------------- Plots for differential analysis -----------------------------#
	#-------------------------------------------------------------------------------------------#

	def plot_heatmap(self, n_rules=20, sort_by="confidence_log2fc", color_by="lift_log2fc"):
		"""
		"""

		tfcomb.plotting.heatmap()

		"""
		sns.clustermap(np.log(normed[lift_columns] + 0.1), 
               vmax=0.1, 
               row_cluster=False,
               yticklabels=False)
		"""
		#def plot_heatmap(self, n_rules=20, sort_by="lift", color_by="lift", figsize=(8,8)):

		pass

	def plot_bubble(self, n_rules=20, sort_by="confidence_log2fc", yaxis="confidence_log2fc",
																   color_by="lift_log2fc", 
																   size_by="lift_log2fc", figsize=(7,7)):

		"""
		"""

		#Select top/bottom n rules
		sorted_table = self.table.sort_values(sort_by, ascending=False)
		sorted_table.index = sorted_table["TF1"] + " + " + sorted_table["TF2"]

		top_rules = sorted_table.head(n_rules)
		bottom_rules = sorted_table.tail(n_rules)

		# Draw each cell as a scatter point with varying size and color
		fig, (ax1, ax2) = plt.subplots(2, figsize=figsize, constrained_layout=True) 

		tfcomb.plotting.bubble(data=top_rules, yaxis=yaxis, color_by=color_by, size_by=size_by, ax=ax1)
		tfcomb.plotting.bubble(data=bottom_rules, yaxis=yaxis, color_by=color_by, size_by=size_by, ax=ax2)

		#Remove x-axis label for upper plot

		return(fig)