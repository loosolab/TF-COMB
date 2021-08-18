#cython: language_level=3

import numpy as np
cimport numpy as np
import cython


#---------------------------------------------------------------------------------------#
@cython.cdivision(True)		#no check for zero division
@cython.boundscheck(False)	#dont check boundaries
@cython.wraparound(False) 	#dont deal with negative indices
@cython.nonecheck(False)
def count_co_occurrence(np.ndarray[np.int_t, ndim=2] sites, 
						int min_distance = 0,
						int max_distance = 100, 
						float max_overlap = 0, 
						int binary = 0,
						int n_names = 1000):

	"""
	Superfast counting of TF-TF co-occurrences within a given windowsize and with a maximum overlap fraction 
	
	Parameters:
	------------
	sites : np.array
		List of coordinate-lists (chr, start, stop, name) sorted by (chromosom, start)
	w : int
		Windowsize

	min_distance : int
		Minimum allowed distance between two TFs. Default: 0

	max_distance : int
		Maximum allowed distance between two TFs. Default: 100

	max_overlap (float): 
		maximum overlap fraction allowed e.g. 0 = no overlap allowed, 1 = full overlap allowed. Default: 0.

	binary : int
		0 or 1 bool integer. If 0; counts are left raw. If 1; each pair is only counted once per window.

	Returns:
	-----------
	tuple


	"""

	cdef dict count_dict = {} #for saving co-occurrences
	cdef int n_sites = len(sites)
	
	#Create n x n count matrix
	cdef np.ndarray[np.int64_t, ndim=1] TF2_counts = np.zeros(n_names, dtype=int) #for counting TF2 during run
	cdef np.ndarray[np.int64_t, ndim=1] TF2_counts_adjustment = np.zeros(n_names, dtype=int) #for counting TF2 binary adjustments during run
	cdef np.ndarray[np.int64_t, ndim=1] single_count_arr = np.zeros(n_names, dtype=int)
	cdef np.ndarray[np.int64_t, ndim=2] pair_count_mat = np.zeros((n_names, n_names), dtype=int)
	
	cdef int i = 0
	cdef int j = 0
	cdef int k
	cdef bint finding_assoc = True
	cdef int TF1_chr, TF1_name, TF1_start, TF1_end, 
	cdef int TF2_start, TF2_end, overlap_bp, short_bp, valid_pair
	cdef int TF2_chr, TF2_name
	cdef int distance
	cdef int self_count

	#Loop over all sites
	while i < n_sites: #i is 0-based index, so when i == n_sites, there are no more sites
		
		#Get current TF information
		TF1_chr = sites[i,0]
		TF1_start = sites[i,1]
		TF1_end = sites[i,2]
		TF1_name = sites[i,3]

		#Count TF1
		single_count_arr[TF1_name] += 1

		#Initialize array to 0 for counting TF2-counts
		for k in range(n_names):
			TF2_counts[k] = 0
			TF2_counts_adjustment[k] = 0

		self_count = 0 #count number of times a TF1-TF1 pair was found in window

		#Find possible associations with TF1 within window 
		finding_assoc = True
		j = 0
		while finding_assoc == True:
			
			#Next site relative to TF1
			j += 1
			if j+i >= n_sites: #j+i index site is beyond end of list, increment i
				i += 1
				finding_assoc = False #break out of finding_assoc

			else:	#There are still sites available

				#Fetch information on TF2-site
				TF2_chr = sites[i+j,0]
				TF2_start = sites[i+j,1]
				TF2_end = sites[i+j,2]
				TF2_name = sites[i+j,3]
				
				#Calculate distance between the two sites
				distance = TF2_start - TF1_end #TF2_start - TF1_end will be negative if TF1 and TF2 are overlapping
				if distance < 0:
					distance = 0

				if (TF1_chr == TF2_chr) and (distance <= max_distance): #check that sites are within window
					if distance >= min_distance: #Check that sites are more than min distance away
						
						# check if they are overlapping more than the threshold
						valid_pair = 1
						if distance == 0:	#distance is 0 if the sites are overlapping or book-ended
							overlap_bp = TF1_end - TF2_start #will be negative if no overlap is found
							
							# Get the length of the shorter TF
							short_bp = min([TF1_end - TF1_start, TF2_end - TF2_start])
							
							#Invalid pair, overlap is higher than threshold
							if overlap_bp / (short_bp*1.0) > max_overlap:  #if overlap_bp is negative; this will always be False
								valid_pair = 0

						#Save counts of association
						if valid_pair == 1:
							
							TF2_counts[TF2_name] += 1
							TF2_counts_adjustment[TF2_name] += self_count

							#Count TF1 self-counts for adjusting to binary flag
							if binary == 1 and TF1_name == TF2_name:
								self_count += 1
			
				else:
					#The next site is out of window range; increment to next i
					i += 1
					finding_assoc = False   #break out of finding_assoc-loop

		## Done finding TF2's for current TF1
		
		#Should counts be binarized?
		if binary == 1:
			for k in range(n_names):

				#Convert all TF1-TF2 counts above 1 -> 1
				if TF2_counts[k] > 1:
					TF2_counts[k] = 1

				if TF2_counts_adjustment[k] > 1:
					TF2_counts_adjustment[k] = 1

				#Adjust for multiple TF1 within each window
				TF2_counts[k] -= TF2_counts_adjustment[k] #counts are removed due to adjustment

		#Add counts to pair_count_mat
		for k in range(n_names):
			pair_count_mat[TF1_name, k] += TF2_counts[k] 

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

def count_distances(np.ndarray[np.int_t, ndim=2] sites,
					np.ndarray[np.int_t, ndim=2] rules,
					int min_distance = 0,
					int max_distance = 100, 
					short anchor_mode = 0):

	"""
	Superfast counting of TF-TF co-occurrences within a given windowsize and with a maximum overlap fraction 
	
	Parameters:
	------------
	sites : np.ndarray
		List of coordinate-lists (chr, start, stop, name) sorted by (chromosom, start)
	
	rules : np.ndarray
		List of pairs (tf1 name, tf2 name) encoded as int

	min_distance : int
		Minimum allowed distance between two TFs. Default: 0

	max_distance : int
		Maximum allowed distance between two TFs. Default: 100

	anchor_mode : short
		anchor mode to calculate distance with. One of [0,1,2]. 
		0 = inner, 1 = outer, 2 = center. Default: 0

	Returns:
	-----------
	dist_count_mat: np.ndarray
		n x (distance range) matrix. 


	"""

	cdef int n_sites = len(sites)
	
	cdef list pairs = list()
	cdef np.ndarray[np.int64_t, ndim=1] rule

	#Create n x distance range matrix || +3 for 2 tf names + 1-off
	cdef np.ndarray[np.int64_t, ndim=2] dist_count_mat = np.zeros((len(rules), 3+(max_distance-min_distance)), dtype=int)

	cdef int i = 0
	cdef int j = 0
	cdef bint finding_assoc = True
	cdef int TF1_chr, TF1_name, TF1_start, TF1_end, TF1_anchor 
	cdef int TF2_start, TF2_end, TF2_anchor, valid_pair
	cdef int TF2_chr, TF2_name
	cdef int distance
	cdef int pair_ind = 0
	
	# initialize tfnames
	for rule in rules:
		pairs.append([rule[0],rule[1]])
		dist_count_mat[ind, 0] = rule[0]
		dist_count_mat[ind, 1] = rule[1]
		ind += 1
	#Loop over all sites
	while i < n_sites: #i is 0-based index, so when i == n_sites, there are no more sites
		#Get current TF information
		TF1_chr = sites[i,0]
		TF1_start = sites[i,1]
		TF1_end = sites[i,2]
		TF1_name = sites[i,3]

		#Find possible associations with TF1 within window 
		finding_assoc = True
		j = 0
		while finding_assoc == True:
			#Next site relative to TF1
			j += 1
			if j+i >= n_sites: #j+i index site is beyond end of list, increment i
				i += 1
				finding_assoc = False #break out of finding_assoc

			else:	#There are still sites available

				#Fetch information on TF2-site
				TF2_chr = sites[i+j,0]
				TF2_start = sites[i+j,1]
				TF2_end = sites[i+j,2]
				TF2_name = sites[i+j,3]
				
				# Check anchor mode 	
    			# 1 = outer
				if (anchor_mode == 1): 
					TF1_anchor = TF1_start
					TF2_anchor = TF2_end
				# 2 = center
				elif (anchor_mode == 2):
					TF1_anchor = int(np.ceil((TF1_end - TF1_start) / 2))
					TF2_anchor = int(np.ceil((TF2_end - TF2_start) / 2))
				# 0 = inner (default)
				else:
					TF1_anchor = TF1_end
					TF2_anchor = TF2_start

				#Calculate distance between the two sites
				distance = TF2_anchor - TF1_anchor #TF2_start - TF1_end will be negative if TF1 and TF2 are overlapping
				if (TF1_chr == TF2_chr) and (distance <= max_distance): #check that sites are within window
					if distance >= min_distance: #Check that sites are more than min distance away
						valid_pair = 0
						pair_ind = -1

						if [TF1_name,TF2_name] in pairs:
							valid_pair = 1
							pair_ind = pairs.index([TF1_name,TF2_name])


						#Save counts of association
						if valid_pair == 1:
							# min_distance is offset (negative min distance adds up, positive decrease the index) || +2 for TF names
							dist_count_mat[pair_ind, (distance - min_distance)+2] += 1
			
				else:
					#The next site is out of window range; increment to next i
					i += 1
					finding_assoc = False   #break out of finding_assoc-loop

		## Done finding TF2's for current TF1
		

	return (dist_count_mat)