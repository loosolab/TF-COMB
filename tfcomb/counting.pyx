#cython: language_level=3

import numpy as np
cimport numpy as np
import cython

#---------------------------------------------------------------------------------------#
@cython.cdivision(True)		#no check for zero division
@cython.boundscheck(False)	#dont check boundaries
@cython.wraparound(False) 	#dont deal with negative indices
@cython.nonecheck(False)
def count_co_occurrence(np.ndarray[np.int_t, ndim=2] sites, int w, int max_overlap, int n_names = 1000):

	"""
	Superfast counting of TF-TF co-occurrences within a given windowsize and with a maximum overlap fraction 
	
	Parameters:
	------------
	sites : np.array
		List of coordinate-lists (chr, start, stop, name) sorted by (chromosom, start)
	w : int
		Windowsize
		
	max_overlap (float)
		maximum overlap fraction allowed e.g. 0 = no overlap allowed, 1 = full overlap allowed.
	output: count_dict   - Dictionary with all TFs and all occuring TF pairs with corresponding count

	Returns:
	-----------


	"""

	cdef dict count_dict = {} #for saving co-occurrences
	cdef int n_sites = len(sites)
	
	#Create n x n count matrix
	cdef np.ndarray[np.int64_t, ndim=1] single_count_arr = np.zeros(n_names, dtype=int)
	cdef np.ndarray[np.int64_t, ndim=2] pair_count_mat = np.zeros((n_names, n_names), dtype=int)

	cdef int i = 0
	cdef int j = 0
	cdef bint finding_assoc = True
	cdef int TF1_chr, TF1_name, TF1_start, TF1_end, 
	cdef int TF2_start, TF2_end, overlap_bp, short_bp, valid_pair
	cdef int TF2_chr, TF2_name

	#Loop over all sites
	while i < n_sites: #i is 0-based index, so when i == n_sites, there are no more sites
		
		#Get current TF information
		TF1_chr = sites[i,0]
		TF1_start = sites[i,1]
		TF1_end = sites[i,2]
		TF1_name = sites[i,3]

		#Count TF1
		single_count_arr[TF1_name] += 1

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
				TF2_chr = sites[i+j,0]
				TF2_start = sites[i+j,1]
				TF2_end = sites[i+j,2]
				TF2_name = sites[i+j,3]

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
						pair_count_mat[TF1_name,TF2_name] += 1
			
				else:
					#The next site is out of window range; increment to next i
					i += 1
					finding_assoc = False   #break out of finding_assoc-loop
	
	return((single_count_arr, pair_count_mat))

def get_unique_bp(np.ndarray[np.int_t, ndim=2] sites):
	"""

	"""

	cdef int n_sites = len(sites)
	cdef int total_bp = 0
	cdef int current_chrom, current_start, current_end

	cdef int previous_start = sites[0,1]
	cdef int previous_end = sites[0,2]
	cdef int i = 1

	#Loop over all sites
	while i < n_sites: #i is 0-based index, so when i == n_sites, there are no more sites

		current_chrom = sites[i,0]
		current_start = sites[i,1]
		current_end = sites[i,2]

		if current_start > previous_end:
			#Gap; add to total_bp
			stretch_bp = previous_end - previous_start 

			previous_start = current_start
			previous_end = current_end

		else:
			previous_end = current_end

		#current_end = "" #sites[]
		#previous = ""

		i += 1
	
	#Add last region
	total_bp += previous_end - previous_start

	return(total_bp)
