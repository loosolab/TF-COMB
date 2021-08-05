from kneed import DataGenerator, KneeLocator
from tobias.utils.regions import OneRegion, RegionList
from tobias.utils.motifs import MotifList
from copy import deepcopy
import pandas as pd
import statsmodels.stats.multitest
import numpy as np
import scipy.stats
import pysam

def check_columns(df, columns):
	""" Utility to check whether columns are found within df """
	
	df_columns = df.columns

	for column in columns:
		if column is not None:
			if column not in df_columns:
				raise ValueError("Column '{0}' is not found in dataframe. Available columns are: {1}".format(column, df_columns))

def check_type(obj, types):
	"""
	Parameters:
	----------
	obj : object
		Object to check type on
	types : list
		A list of allowed object types
	"""
	
	#Check if any of the types fit
	flag = 0
	for t in types:
		if isinstance(obj, t):
			flag = 1

	#Raise valueError if none of the types fit
	if flag == 0:	
		raise ValueError("Object has type '{0}', but must be one of: {1}".format(type(obj), types))


def _get_background(matrix, TF1_idx, TF2_idx):
	""" Fetches the background values from a matrix for a particular set of TFs"""

	#Collect values for background
	TF1_background = matrix[TF1_idx,:]
	TF1_background = np.delete(TF1_background, TF2_idx, 0) #exclude TF1-TF2

	TF2_background = matrix[:,TF2_idx]
	TF2_background = np.delete(TF2_background, TF1_idx, 0) #exclude TF1-TF2

	background = list(TF1_background) + list(TF2_background)

	return(background)


def prepare_motifs(motifs_file, motif_pvalue=0.0001, motif_naming="name"):
	""" Read """

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
	RegionList of TFBS within regions

	"""

	#open the genome given
	if isinstance(genome, str):
		genome_obj = open_genome(genome)
	else:
		genome_obj = genome

	TFBS = RegionList()

	for region in regions:
		seq = genome_obj.fetch(region.chrom, region.start, region.end)
		region_TFBS = motifs.scan_sequence(seq, region)
		region_TFBS.loc_sort()

		TFBS += region_TFBS

	if isinstance(genome, str):
		genome_obj.close()

	return(TFBS)

def merge_regions(regions):
	"""	Merge overlapping coordinates within regions """

	regions.loc_sort()
	merged = []
	for region in regions:
		pass

	return(merged)


def remove_duplicates(TFBS):
	""" """

	filtered = TFBS

	return(filtered)


def resolve_overlapping(TFBS):
	""" Remove self overlapping regions """

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


def _calculate_pvalue(table, measure="cosine", alternative="greater"):
	"""
	Calculates the p-value of each TF1-TF2 pair for the measure given.

	Parameters
	------------
	table : pd.DataFrame
		The table from '.rules' of DiffObj or DiffCombObj
	measure : str
		The measure to calculate pvalue for
	alternative : str
		One of: 'two-sided', 'greater', 'less'

	Returns
	--------
	List of p-values in order of input table

	"""
	
	pivot_table = pd.pivot(table, index="TF1", columns="TF2", values=measure)    
	
	#Fill NA with 0's
	pivot_table.fillna(0, inplace=True)
	
	#Convert to matrix
	matrix = pivot_table.to_numpy()
	TF1_list = pivot_table.index.tolist()
	TF2_list = pivot_table.columns.tolist()
	
	TF1_dict = {TF1: TF1_list.index(TF1) for TF1 in TF1_list}
	TF2_dict = {TF2: TF2_list.index(TF2) for TF2 in TF2_list}
	
	TF1_list_int = table["TF1"].replace(TF1_dict)
	TF2_list_int = table["TF2"].replace(TF2_dict)

	combinations = list(zip(TF1_list_int, TF2_list_int))
	n_pairs = len(combinations)
	pvalues = [1] * n_pairs
	
	for i, (TF1, TF2) in enumerate(combinations):

		#if i % 100 == 0:
			#print(i)

		#Collect values for pair and background
		obs = matrix[TF1, TF2]
		background = _get_background(matrix, TF1, TF2)

		#Calculate pvalue 
		mu = np.mean(background)
		std = np.std(background)
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
			
		pvalues[i] = p

	return(pvalues)

	#Adjust pvalues
	#table["pvalue_adj"] = statsmodels.stats.multitest.multipletests(table["pvalue"], method="Bonferroni")[1]

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


def assign_sites_to_regions(sites, regions):
	"""
	Assign sites to each region by overlapping. Used for the tfcomb.ComObj.count_between() function to assign .TFBS to regions.

	Parameters
	-----------
	sites : tobias.utils.RegionList()
		Individual sites to assign to larger regions
	regions : tobias.utils.RegionList()
		RegionList of regions for which to assign sites

	Returns
	-------
	dict 
		Dictionary of format {(chr,start,stop}: RegionList(<site1>, <site2>) (...)}
	"""
	
	if not isinstance(sites, RegionList):
		raise ValueError("sites")

	if not isinstance(regions, RegionList):
		raise ValueError("regions")

	#if regions == str
		#open regions as RegionList
	
	#Ensure that input is location sorted
	regions = deepcopy(regions)
	regions.loc_sort()

	sites = deepcopy(sites)
	sites.loc_sort()

	#Get all chromosomes in regions and TFBS in order
	site_chroms = []
	for site in sites:
		if site.chrom not in site_chroms:
			site_chroms.append(site.chrom)
	#print("Chromosomes for sites: {0}".format(site_chroms))

	region_chroms = []
	for region in regions:
		if region.chrom not in region_chroms:
			region_chroms.append(region.chrom)
	#print("Chromosomes for regions: {0}".format(region_chroms))

	#Match site to regions
	sites_in_regions = {} 	#dictionary of style {(chr1, 0, 100): RegionList(<sites>)}
	n_sites = len(sites)
	n_regions = len(regions)

	site_i = 0
	reg_i = 0
	while (reg_i < n_regions) and (site_i < n_sites):
			
			current_region = regions[reg_i]
			current_site = sites[site_i]
			#print("current_region: {0} | current_site: {1}".format(current_region, current_site))

			if current_region.chrom != current_site.chrom: #Chromosomes are different
				#print("Different chroms")

				#Is TFBS chrom in region chroms?
				if current_site.chrom in region_chroms:

					#Find out which list to increment 
					reg_chrom_idx = region_chroms.index(current_region.chrom)
					site_chrom_idx = region_chroms.index(current_site.chrom)

					if reg_chrom_idx > site_chrom_idx: #If the region has higher idx than TFBS; increment TFBS
						site_i += 1
					else: #if region has lower idx than TFBS; increment region
						reg_i += 1
				else:
					current_site += 1 #increment TFBS to find potential TFBS in region_chroms
				
			else: #on same chromosome; find overlaps
				#print("Finding possible overlaps...")

				if current_site.end <= current_region.end: #TFBS is before or within current region
					if current_site.start >= current_region.start: #TFBS is within region; save
						region_tup = (current_region.chrom, current_region.start, current_region.end)

						#Initialize region
						if region_tup not in sites_in_regions:
							sites_in_regions[region_tup] = RegionList()

						sites_in_regions[region_tup].append(current_site)

						site_i += 1 #increment TFBS 
						
					else:	#TFBS is before region; increment TFBS
						site_i += 1 

				else: #TFBS is after current region; increment regions
					#print("{0} is after {1} - increment regions".format(current_site, current_region))
					reg_i += 1

	return(sites_in_regions)

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
