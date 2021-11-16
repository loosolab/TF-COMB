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
						int anchor = 0,
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
	anchor : int
		Anchor used to calculate distance with. One of [0,1,2] (0 = inner, 1 = outer, 2 = center). Default: 0
	n_names : int
		Number of unique names within sites. Used to initialize the count arrays.

	Returns:
	-----------
	tuple
		(np.array 1 x n_names, np.array(n_names x n_names))

	"""

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
	cdef int TF2_chr, TF2_name, TF2_start, TF2_end
	cdef int overlap_bp, short_bp, valid_pair
	cdef int TF1_anchor, TF2_anchor
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
				
				#Calculate distance between the two sites based on anchor
				if (anchor == 1): # 1 = outer
					TF1_anchor = TF1_start
					TF2_anchor = TF2_end
				elif (anchor == 2): # 2 = center
					TF1_anchor = (TF1_start + TF1_end) / 2
					TF2_anchor = (TF2_start + TF2_end) / 2
				else: #0 = inner (default)
					TF1_anchor = TF1_end
					TF2_anchor = TF2_start					
					
				#Calculate distance between the two sites
				distance = TF2_anchor - TF1_anchor #TF2_start - TF1_end will be negative if TF1 and TF2 are overlapping
				if distance < 0:
					distance = 0 #cap any negative distances to 0

				#Check if sites are valid as co-occurring
				#valid_pair = 0 #initialize as non-valid
				if (TF1_chr == TF2_chr) and (distance <= max_distance):
					if distance >= min_distance: #check that sites are within window; else we stay in finding_assoc and increment j
						valid_pair = 1

						#Check if sites overlap more than threshold
						short_bp = min([TF1_end - TF1_start, TF2_end - TF2_start]) # Get the length of the shorter TF
						overlap_bp = TF1_end - TF2_start #will be negative if no overlap is found
						if overlap_bp > short_bp: #overlap_bp can maximally be the size of the smaller TF (is larger when TF2 is completely within TF1)
							overlap_bp = short_bp

						#Invalid pair if overlap is higher than threshold
						if (overlap_bp / (short_bp*1.0)) > max_overlap:  #if overlap_bp is negative; this will always be False
							valid_pair = 0

						#Save counts of association
						if valid_pair == 1:
							TF2_counts[TF2_name] += 1
							TF2_counts_adjustment[TF2_name] += self_count

							#Count TF1 self-counts for adjusting to binary flag
							if binary == 1 and TF1_name == TF2_name:
								self_count += 1

				elif (TF1_chr == TF2_chr) and (distance > max_distance): #This TF2 is on the same chromosome but more than max_distance away
					
					#Establish if all valid sites were found for TF1
					if anchor == 0: #inner distance

						#The next site is out of inner window range; increment to next i
						i += 1
						finding_assoc = False   #break out of finding_assoc-loop
					
					else: #If anchor is outer or center, there might still be valid pairs for future TF2's

						#Check if it will be possible to find valid pairs in next sites
						if TF2_start > TF1_start + max_distance:
							#no longer possible to find valid pairs for TF1; increment to next i
							i += 1
							finding_assoc = False   #break out of finding_assoc-loop

				elif TF1_chr != TF2_chr: #TF2 is on another chromosome
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

@cython.cdivision(True)		#no check for zero division
@cython.boundscheck(False)	#dont check boundaries
@cython.wraparound(False) 	#dont deal with negative indices
@cython.nonecheck(False)
def count_distances(np.ndarray[np.int_t, ndim=2] sites,
					dict rules,
					int min_distance = 0,
					int max_distance = 100,
					float max_overlap = 0, 
					short anchor_mode = 0, 
					bint directional = False):

	"""
	Superfast counting of TF-TF co-occurrences within a given windowsize and with a maximum overlap fraction 
	
	Parameters:
	------------
	sites : np.ndarray
		List of coordinate-lists (chr, start, stop, name) sorted by (chromosom, start)
	rules :dict
		dict of pairs (tf1 name, tf2 name): index with tf1 name, tf2 name encoded as int
	min_distance : int
		Minimum allowed distance between two TFs. Default: 0
	max_distance : int
		Maximum allowed distance between two TFs. Default: 100
	max_overlap (float): 
		maximum overlap fraction allowed e.g. 0 = no overlap allowed, 1 = full overlap allowed. Default: 0.
	anchor_mode : short
		anchor mode to calculate distance with. One of [0,1,2]. 
		0 = inner, 1 = outer, 2 = center. Default: 0
	directional : bool
		Whether to count TF1-TF2 exclusively, or count TF2-TF1 counts for the TF1-TF2 pair. Setting 'False' means TF1-TF2 counts == TF2-TF1 counts, and
		'True' means distances are only counted in the TF1-TF2 direction. Default: False.

	Returns:
	-----------
	dist_count_mat: np.ndarray
		n x (distance range) matrix. 


	"""
	
	#Intialize cython vars
	cdef int n_sites = len(sites)
	cdef int i = 0
	cdef int j = 0
	cdef bint finding_assoc = True
	cdef int TF1_chr, TF1_name, TF1_start, TF1_end
	cdef int TF2_chr, TF2_name, TF2_start, TF2_end
	cdef int TF1_anchor, TF2_anchor
	cdef int valid_pair, overlap_bp, short_bp
	
	cdef int distance
	cdef int pair_ind = 0
	cdef int ind = 0

	#Count all negative distances as "-1"
	if min_distance == 0 and max_overlap > 0:
		min_distance = -1

	#Create rule x distance range matrix || +3 for 2 tf names + 1-off 
	cdef int offset = 3
	cdef np.ndarray[np.int64_t, ndim=2] dist_count_mat = np.zeros((len(rules), offset + (max_distance-min_distance)), dtype=int)
	
	# initialize tfnames
	for tf1, tf2 in rules:
		dist_count_mat[ind, 0] = tf1
		dist_count_mat[ind, 1] = tf2
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
				
				#Check if TF1_name,TF2_name is a target rule; otherwise, do not count
				if (TF1_name, TF2_name) in rules or (TF2_name, TF1_name) in rules: #count both TF1-TF2 and TF2-TF1 directions

					# Check anchor mode 	
					if (anchor_mode == 1): # 1 = outer
						TF1_anchor = TF1_start
						TF2_anchor = TF2_end
					elif (anchor_mode == 2): # 2 = center
						TF1_anchor = (TF1_start + TF1_end) / 2
						TF2_anchor = (TF2_start + TF2_end) / 2
					else: # 0 = inner (default)
						TF1_anchor = TF1_end
						TF2_anchor = TF2_start

					#Calculate distance between the two sites
					distance = TF2_anchor - TF1_anchor #TF2_start - TF1_end will be negative if TF1 and TF2 are overlapping
					if distance < 0:
						distance = -1 #cap any negative distances to -1

					if (TF1_chr == TF2_chr) and (distance <= max_distance): #check that sites are within window
						if distance >= min_distance: #Check that sites are more than min distance away; else stay in finding_assoc
							valid_pair = 1 #so-far, the pair is valid
							
							#Calculate potential overlap between TF1/TF2
							short_bp = min([TF1_end - TF1_start, TF2_end - TF2_start])

							#Calculate overlap between TF1/TF2
							overlap_bp = TF1_anchor - TF2_anchor #will be negative if no overlap is found
							if overlap_bp > short_bp: #overlap_bp can maximally be the size of the smaller TF (is larger when TF2 is completely within TF1)
								overlap_bp = short_bp

							#Invalid pair, overlap is higher than threshold
							if (overlap_bp / (short_bp*1.0)) > max_overlap:  #if overlap_bp is negative; this will always be False
								valid_pair = 0

							#Save counts of association
							if valid_pair == 1:
								
								#dist_ind calculation remains the same regardless directionality
								dist_ind = offset + distance - min_distance - 1 
								
								#Add counts for the observed TF1-TF2
								if (TF1_name, TF2_name) in rules:
									pair_ind = rules[(TF1_name, TF2_name)] #index of pair in count mat
									dist_count_mat[pair_ind, dist_ind] += 1
								
								#Also add counts to TF2-TF1 if not directional
								if directional == False and TF1_name != TF2_name: #do not add extra counts for TF-self-pairs
									if (TF2_name, TF1_name) in rules:
										
										pair_ind = rules[(TF2_name, TF1_name)] #index of pair in count mat
										dist_count_mat[pair_ind, dist_ind] += 1

					elif (TF1_chr == TF2_chr) and (distance > max_distance): #This TF2 is on the same chromosome but more than max_distance away
					
						#Establish if all valid sites were found for TF1
						if anchor_mode == 0: #inner distance

							#The next site is out of inner window range; increment to next i
							i += 1
							finding_assoc = False   #break out of finding_assoc-loop
						
						else: #If anchor is outer or center, there might still be valid pairs for future TF2's

							#Check if it will be possible to find valid pairs in next sites
							if TF2_start > TF1_start + max_distance:
								#no longer possible to find valid pairs for TF1; increment to next i
								i += 1
								finding_assoc = False   #break out of finding_assoc-loop

					else:
						#The next site is out of window range; increment to next i
						i += 1
						finding_assoc = False   #break out of finding_assoc-loop

		## Done finding TF2's for current TF1; i has been incremented

	return (dist_count_mat)