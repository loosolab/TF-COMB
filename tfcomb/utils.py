from kneed import DataGenerator, KneeLocator

def find_knee(x, y):
	""" Find knee in knee-plot using x,y coordinates """
	knee = ""

	return(knee)

def make_symmetric(matrix):
	"""
	Make a numpy matrix matrix symmetric by merging x-y and y-x
	"""
	matrix_T = matrix.T 
	symmetric = matrix + matrix_T
	return(symmetric)


def assign_sites_to_regions(sites, regions):
	"""
	Which indices in  are fitting to each region
	#split regions into dictionary of regions
	Used for the count_between

	Parameters
	-----------
	sites : tobias.utils.RegionList()

	regions : tobias.utils.RegionList()
		

	Returns
	-------
	dict 
		Dictionary of format {(chr,start,stop}: RegionList(<TFBS>) (...)}

	"""
	
	if not isinstance(sites, RegionList):
		raise ValueError("sites")
	if not isinstance(regions, RegionList):
		raise ValueError("regions")
	#if regions == str
		#open regions as RegionList
	
	#Ensure that input is location sorted
	regions = regions.copy()
	regions.loc_sort()

	sites = sites.copy()
	sites.loc_sort()

	#Get all chromosomes in regions and TFBS in order
	site_chroms = []
	for site in sites:
		if site.chrom not in site_chroms:
			site_chroms.append(site.chrom)
	
	region_chroms = []
	for site in regions:
		if site.chrom not in region_chroms:
			region_chroms.append(site.chrom)

	#Match site to regions
	sites_in_regions = {} 	#dictionary of style {(chr1, 0, 100): RegionList(<sites>)}
	n_TFBS = len(sites)
	n_regions = len(regions)

	site_i = 0
	reg_i = 0
	while reg_i < n_regions:
		while site_i < n_TFBS:
			
			current_region = regions[reg_i]
			current_TFBS = TFBS[site_i]

			if current_region.chrom != current_TFBS: #Chromosomes are different
				
				#Is TFBS chrom in region chroms?
				if current_TFBS.chrom in region_chroms:

					#Find out which list to increment 
					reg_chrom_idx = region_chroms.index(current_region.chrom)
					tfbs_chrom_idx = region_chroms.index(current_TFBS.chrom)

					if reg_chrom_idx > tfbs_chrom_idx: #If the region has higher idx than TFBS; increment TFBS
						site_i += 1
					else: #if region has lower idx than TFBS; increment region
						reg_i += 1
				else:
					current_TFBS += 1 #increment TFBS to find potential TFBS in region_chroms
				
			else: #on same chromosome; find overlaps
				if current_TFBS.end <= current_region.end: #TFBS is before or within current region
					if current_TFBS.start >= current_region.start: #TFBS is within region; save
						region_tup = current_region.tup()
						if region_tup not in sites_in_regions:
							sites_in_regions[region_tup] = RegionList()
						sites_in_regions[region_tup].append(current_TFBS)

						site_i += 1 #increment TFBS 
						
					else:	#TFBS is before region; increment TFBS
						site_i += 1 

				else: #TFBS is after current region; increment regions
					reg_i += 1

	return(sites_in_regions)