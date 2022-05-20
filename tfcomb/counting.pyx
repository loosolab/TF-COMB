#cython: language_level=3

import numpy as np
cimport numpy as np
import cython


#---------------------------------------------------------------------------------------#
def rolling_mean(np.ndarray[np.float64_t, ndim=1] arr, int w):
	"""
	Rolling mean of arr with window size w. The mean of the border regions are calculated on the reduced window, 
	e.g. with a window size of 3, the first and last windows only consists of 2 values.
	"""

	cdef int L = arr.shape[0]
	cdef np.ndarray[np.float64_t, ndim=1] mean_arr = np.zeros(L)   #mean of values per window
	
	cdef int i, j, w_start, w_end, n_vals
	cdef int lf = int(np.floor((w - 1) / 2.0))
	cdef int rf = int(np.ceil((w - 1)/ 2.0))
	cdef float valsum

	#Mean per window
	for i in range(L):
		
		#Ranges of window
		w_start = max(0, i-lf)
		w_end = min(i+rf+1, L) #w_end includes +1

		#Sum in window
		valsum = 0
		for j in range(w_start, w_end):
			valsum += arr[j]
		n_vals = w_end - w_start

		#Mean of window
		mean_arr[i] = valsum / (n_vals*1.0)

	return mean_arr


#---------------------------------------------------------------------------------------#
@cython.cdivision(True)		#no check for zero division
@cython.boundscheck(False)	#dont check boundaries
@cython.wraparound(False) 	#dont deal with negative indices
@cython.nonecheck(False)
def count_co_occurrence(np.ndarray[np.int_t, ndim=2] sites, 
						int min_dist=0,
						int max_dist=100, 
						float min_overlap=0,
						float max_overlap=0, 
						bint binarize=False,
						int anchor=0, 
						int n_names=1000,
						bint directional=False,
						int task=1,
						list rules=[],
						bint percentage=False,
						int percentage_bins=100					
						):

	"""
	Superfast counting of TF-TF co-occurrences within a given min/max distance and with a maximum overlap fraction 
	
	Parameters:
	------------
	sites : np.array
		Array of coordinate-lists (chr, start, stop, name) sorted by (chromosom, start).
	min_dist : int, optional
		Minimum allowed distance between two TFs. Default: 0.
	max_dist : int, optional
		Maximum allowed distance between two TFs. Default: 100.
	min_overlap : float, optional
		Minimum overlap fraction needed between sites, e.g. 0 = no overlap needed, 1 = full overlap needed. Default: 0.
	max_overlap : float, optional
		Maximum overlap fraction allowed between sites e.g. 0 = no overlap allowed, 1 = full overlap allowed. Default: 0.
	binarize : bool, optional
		If False; TF1-TF2 can be counted more than once for the same TF1 (if TF2 occurs multiple times per window). If True; a TF1-TF2 pair is only counted once per TF1 occurrence. Default: False.
	anchor : int, optional
		Anchor used to calculate distance with. One of [0,1,2] (0 = inner, 1 = outer, 2 = center). Default: 0
	n_names : int, optional
		Number of unique names within sites. Used to initialize the count arrays. Default: 1000
	task : int, optional
		Which task to perform:
		If 1; count number of co-occurring pairs beween all input sites. 
		If 2; count the distribution of distances per pair given in 'rules'. 
		If 3; get the indices of co-occurring pairs for the pairs given in 'rules'. 
		Default: 1.
	directional : bool, optional
		For task == 2, 'directional' controls whether to count TF1-TF2 exclusively, or count TF2-TF1 counts for the TF1-TF2 pair. Setting 'False' means TF1-TF2 counts == TF2-TF1 counts, and
		'True' means distances are only counted in the TF1-TF2 direction. Default: False.		
	rules : list, optional
		The rules to be taken into account. Is only used for task == 2 and task == 3. Must be a list of tuple-pairs (TF1_name, TF2_name) encoded as int.
	percentage : bool, optional
		For task == 2; whether to count distances as base pairs or percentage of longest TF length. Default: False.
	percentage_bins : int, optional
		For task == 2 and dist_percentage == True; how many bins for collecting distances. Default: 101 (one per percent + 0).
	
	Returns:
	-----------
	If task == 1:
		(np.array(1 x n_names), np.array(n_names x n_names))
	If task == 2:
		np.array(n_rules x 2 + min_dist:max_dist) 
	If task == 3:
		np.array( <n_pairs> x 2)
	"""

	cdef int n_sites = len(sites)
	
	#Create n x n count matrix
	cdef np.ndarray[np.int64_t, ndim=1] TF2_counts = np.zeros(n_names, dtype=int) #for counting TF2 during run
	cdef np.ndarray[np.int64_t, ndim=1] TF2_counts_adjustment = np.zeros(n_names, dtype=int) #for counting TF2 binary adjustments during run
	cdef np.ndarray[np.int64_t, ndim=1] single_count_arr = np.zeros(n_names, dtype=int)
	cdef np.ndarray[np.int64_t, ndim=2] pair_count_mat = np.zeros((n_names, n_names), dtype=int)
	
	#Declare variables
	cdef int i = 0
	cdef int j = 0
	cdef int k
	cdef bint finding_assoc = True
	cdef bint valid_pair
	cdef int TF1_chr, TF1_start, TF1_end, TF1_name
	cdef int TF2_chr, TF2_start, TF2_end, TF2_name
	cdef int overlap_bp, short_bp, long_bp
	cdef int TF1_anchor, TF2_anchor
	cdef int distance
	cdef int self_count
	cdef double overlap_frac
	
	#Initializations for distance counting (not necessarily used)
	cdef int n_pairs = len(rules)
	cdef int rule_idx
	cdef int dist_idx
	cdef int ind = 1 

	if task == 2 and percentage == True:
		n_distances = 2+percentage_bins+1 #names and bins (+0)
	else:
		n_distances = 2+max_dist-min_dist+1 #+1 to make range inclusive
		
	cdef np.ndarray[np.int64_t, ndim=2] dist_count_mat = np.zeros((n_pairs+1, n_distances), dtype=int) #2 first cols are names, then min_dist:max dist both included (+1)
	cdef np.ndarray[np.int64_t, ndim=2] dist_indices_mat = np.zeros((n_names, n_names), dtype=int) #0 index is reserved for non-counted pairs

	#Initializations for pair locations (not necessarily used)
	cdef int loc_rows = 10000 #initialize with 10000 rows for efficiency
	cdef int loc_idx = 0 #current index in pair_locations_mat
	cdef np.ndarray[np.int64_t, ndim=2] pair_locations_mat = np.zeros((loc_rows, 2), dtype=int)

	#Setup index matrix for counting distances (if chosen)
	if task == 2 or task == 3:
		for (TF1_name, TF2_name) in rules:
			dist_indices_mat[TF1_name, TF2_name] = ind #ind initializes at 1

			dist_count_mat[ind, 0] = TF1_name 	#first column is TF1 name
			dist_count_mat[ind, 1] = TF2_name 	#second column is TF2 name
			ind += 1

	#Loop over all sites to identify co-occurrences
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
			if i+j >= n_sites: #Si+j index site is beyond end of list, increment i
				i += 1
				finding_assoc = False #break out of finding_assoc

			else:	#There are still sites available
				valid_pair = True #so-far, the pair is valid

				#Fetch information on TF2-site
				TF2_chr = sites[i+j,0]
				TF2_start = sites[i+j,1]
				TF2_end = sites[i+j,2]
				TF2_name = sites[i+j,3]

				#Check if pair is even possible
				if TF1_chr != TF2_chr:
					valid_pair = False
				
				#Check if sites are preliminarily valid as co-occurring
				if valid_pair == True:

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
					distance = TF2_anchor - TF1_anchor #TF2_start - TF1_end (inner) will be negative if TF1 and TF2 are overlapping
					
					if task == 1:
						if distance < 0:
							distance = 0 #cap any negative distances to 0 (overlapping is dist 0)

					#Check if distance is valid:
					if distance <= max_dist:
						if distance >= min_dist: #check that sites are within window; else we stay in finding_assoc and increment j					

							#Check if sites overlap more than threshold
							short_bp = min([TF1_end - TF1_start, TF2_end - TF2_start]) # Get the length of the shorter TF
							overlap_bp = min(TF1_end, TF2_end) - max(TF1_start, TF2_start) #will be negative if no overlap is found
							overlap_bp = max([overlap_bp, 0]) #capped at 0

							#Invalid pair if overlap is higher than threshold
							overlap_frac = (overlap_bp / (short_bp*1.0))
							if (overlap_frac > max_overlap) or (overlap_frac < min_overlap):  #if overlap_bp is negative; this will always be False
								valid_pair = False

							#Save counts of association
							if valid_pair == True: #if pair is still valid
								TF2_counts[TF2_name] += 1
								TF2_counts_adjustment[TF2_name] += self_count

								#Count TF1 self-counts for adjusting to binary flag
								if binarize == 1 and TF1_name == TF2_name:
									self_count += 1

								#Save distance to distribution (if chosen)
								if task == 2:
									
									#Append to distance in dist_count_mat
									rule_idx = dist_indices_mat[TF1_name, TF2_name] #row index of rule (is 0 if pair is not in rules)

									if percentage == True:
										
										long_bp = max([TF1_end - TF1_start, TF2_end - TF2_start]) # Get the length of the longer TF
										
										#Convert any negative distance to positive for percentage
										if distance < 0:
											distance = long_bp + distance 

										#Convert to bin idx
										dist_idx = 2 + <int>((distance * percentage_bins * 1.0) / long_bp) #force to integer
										if dist_idx <= percentage_bins + 2:
											dist_count_mat[rule_idx, dist_idx] += 1

									else:
										dist_idx = 2 + distance - min_dist #column index of distance
										dist_count_mat[rule_idx, dist_idx] += 1

									#Save counts to TF2_name, TF1_name as well
									if directional == False:
										rule_idx = dist_indices_mat[TF2_name, TF1_name] #this can be 0 even if TF1-TF2 is != 0
										
										if percentage == False or (percentage == True and dist_idx <= percentage_bins + 2):
											dist_count_mat[rule_idx, dist_idx] += 1 
										
								#Append indices of pair to list (if chosen)
								if task == 3:
									
									if dist_indices_mat[TF1_name, TF2_name] != 0 or dist_indices_mat[TF2_name, TF1_name] != 0: 
										pair_locations_mat[loc_idx, 0] = i   #TF1
										pair_locations_mat[loc_idx, 1] = i+j #TF2

										loc_idx += 1 #next pair location
										if loc_idx == loc_rows: #next idx would be outside of bounds for pair_locations_mat; add more rows
											empty = np.zeros((10000, 2), dtype=int)
											pair_locations_mat = np.vstack((pair_locations_mat, empty))
											loc_rows += 10000 #loc mat has 10000 more rows

					else: #This TF2 is on the same chromosome but more than max_distance away
						
						#Establish if all valid sites were found for TF1
						if anchor == 0: #inner distance

							#The next site is out of inner window range; increment to next i
							i += 1
							finding_assoc = False   #break out of finding_assoc-loop
						
						else: #If anchor is outer or center, there might still be valid pairs for future TF2's

							#Check if it will be possible to find valid pairs in next sites
							if TF2_start > TF1_anchor + max_dist:
								#no longer possible to find valid pairs for TF1; increment to next i
								i += 1
								finding_assoc = False   #break out of finding_assoc-loop

				else: #TF2 is on another chromosome; stop finding pairs for TF1 at i
					i += 1
					finding_assoc = False   #break out of finding_assoc-loop


		## Done finding TF2's for current TF1
		
		#Should counts be binarized?
		if binarize == 1:
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

	#Return matrices/lists depending on task
	if task == 1:
		return (single_count_arr, pair_count_mat)
	
	elif task == 2:
		dist_count_mat = dist_count_mat[1:,:] #remove the first row (which counted pairs not included in rules)
		return dist_count_mat

	elif task == 3:
		pair_locations_mat = pair_locations_mat[:loc_idx,:] #remove additional empty rows
		return pair_locations_mat
