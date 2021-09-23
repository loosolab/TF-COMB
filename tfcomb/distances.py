from tfcomb.logging import *
from tfcomb.counting import count_distances
from tobias.utils.regions import OneRegion, RegionList
from tobias.utils.signals import fast_rolling_math
from scipy.signal import find_peaks
from tfcomb.logging import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import csv 
import copy 


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
        self._raw = None
        self.distances = None 	     # Numpy array of size n_pairs x maxDist
        self.corrected = None
        self.peaks = None 	         # Numpy array of size n_pairs x n_preferredDistance 
        self.linres = None
        self.normalized = None
        self.peaking_count = None
        self.directional = None
        self.smoothed = None
        self.smooth_window = 3
        self.n_bp = 0			     # Predicted number of baskets 
        self.TFBS = RegionList()     # None RegionList() of TFBS
        self.anchor_mode = 0         # Distance measure mode [0,1,2]
        self.name_to_idx = None      # Mapping TF-names: string <-> int 
        self.pair_to_idx = None      # Mapping Pairs: tuple(string) <-> int
        self.min_dist = 0            # Minimum distance. Default: 0 
        self.max_dist = 300          # Maximum distance. Default: 100
        self.max_overlap = 0         # Maximum overlap. Default: 0       
        self.foldchange_thresh = 1     
        self._PEAK_HEADER = "TF1\tTF2\tDistance\tPeak Heights\tProminences\tProminence Threshold\n"

    def __str__(self):
	    pass
    
    def set_verbosity(self, level):
	    """ Set the verbosity level for logging after creating the CombObj.

		Parameters
		----------
		level : int
			A value between 0-3 where 0 (only errors), 1 (info), 2 (debug), 3 (spam debug). Default: 1.
		"""

	    self.verbosity = level
	    self.logger = TFcombLogger(self.verbosity) #restart logger with new verbosity	    
    
    def fill_rules(self,comb_obj):
        """ Fill object according to reference object 

        Parameters
		----------
		rules : tfcomb.objects
        """
        # TODO: Check instance of combObj (or difcomb)
        self.rules = comb_obj.rules
        # reset pandas index
        self.rules = self.rules.reset_index(drop=True)
        self.TF_names = comb_obj.TF_names
        self.TFBS = comb_obj.TFBS 
        # TODO: min/max dist + overlap

    def set_anchor(self,anchor):
        """ set anchor for distance measure mode
        0 = inner
        1 = outer
        2 = center

        Parameters
		----------
		anchor : one of ["inner","outer","center"]
        """
        #TODO: error if anchor not in modes
        modes = ["inner","outer","center"]
        self.anchor_mode = modes.index(anchor)
		
    def count_distances(self, normalize = True, directional = False):
        """ Count distances for co_occurring TFs, can be followed by analyze_distances
            to determine preferred binding distances
        
        """
        chromosomes = {site.chrom:"" for site in self.TFBS}.keys()
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
        if not directional:
            for i in range(0,self._raw.shape[0]-1):
                if (self._raw[i,0] == self._raw[i+1,1]) and (self._raw[i,1] == self._raw[i+1,0]):
                    s = self._raw[i,2:]+self._raw[i+1,2:]
                    self._raw[i,2:] = s
                    self._raw[i+1,2:] = s
            self.directional = directional
        self._raw_to_human_readable(normalize)

        self.logger.info("Done finding distances! Run .linregress_pair() or .linregress_all() to fit linear regression")
    
    def _raw_to_human_readable(self, normalize = True):
        """ Get the raw distance in human readable format
            
            Returns:
		    ----------
			pd.Dataframe (TF1 name, TF2 name, count min_dist, count min_dist +1, ...., count max_dist)
        """
        self.logger.debug("Converting raw count data to pretty dataframe")
        idx_to_name = {}
        for k,v in self.name_to_idx.items():
            idx_to_name[v] = k 
        
        results = []
        for row in self._raw:
            tf1 = idx_to_name[row[0]]
            tf2 = idx_to_name[row[1]]
            entry = [tf1,tf2]
            self.normalized = normalize
            if normalize:
                entry += (row[2:]/(row[2:].sum())).tolist()
            else:
                entry += row[2:].tolist()
            results.append(entry)
                
        self.distances = pd.DataFrame(results,columns=['TF1','TF2']+[str(x) for x in range (self.min_dist, self.max_dist+1)])

    def linregress_pair(self,pair,n_bins=None, save = None):
        self.logger.debug(f"Fitting linear regression for pair: {pair}")
        if n_bins is None:
            n_bins = self.max_dist - self.min_dist +1
        x = np.linspace(self.min_dist,self.max_dist+1, n_bins)
        #TODO: check pair is valid
        tf1 = pair[0]
        tf2 = pair[1]
        if self.distances is None:
            self.logger.error("No distances evaluated yet. Please run .count_distances() first.")
            sys.exit(0)
        data = self.distances.loc[((self.distances["TF1"]==tf1) &
               (self.distances["TF2"]==tf2))].iloc[0, 2:]
        linres = stats.linregress(range(self.min_dist,self.max_dist+1),np.array(data,dtype = float))
        if save is not None:
            plt.hist(range(self.min_dist,self.max_dist+1),weights=data, bins=n_bins, density=False, alpha=0.6)
            plt.plot(x, linres.intercept + linres.slope*x, 'r', label='fitted line')
            title = f"Fit results: pval = {linres.pvalue},  stderr = {linres.stderr}" 
            plt.title(title)
            plt.savefig(f'{save}linreg_{tf1}_{tf2}.png', dpi=600)
            plt.clf()
        return linres
    
    def linregress_all(self,n_bins = None, save = None):
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
        self.logger.debug(f"Correcting background for pair {pair}")
        if n_bins is None:
            n_bins = self.max_dist - self.min_dist +1
        #TODO: check pair is valid
        tf1 = pair[0]
        tf2 = pair[1]
        if self.distances is None:
            self.logger.error("No distances evaluated yet. Please run .count_distances() first.")
            sys.exit(0)
        data = self.distances.loc[((self.distances["TF1"]==tf1) &
               (self.distances["TF2"]==tf2))].iloc[0, 2:]
        corrected = []
        x_val = 0
        if linres is None:
            self.logger.error("Please fit a linear regression first. [.linregress_all() or .linregress_pair()]")
            sys.exit(0)
        if  not isinstance(linres, stats._stats_mstats_common.LinregressResult):
            self.logger.error("linres need to be a valid scipy LinregressResult type. Use .linregress_all() or .linregress_pair() to create one.")
            sys.exit(0)
        
        for dist in data:
            corrected.append(dist-(linres.intercept + linres.slope*x_val))
            x_val += 1

        if save is not None:
            x = np.linspace(self.min_dist,self.max_dist+1, n_bins)
            plt.hist(range(self.min_dist,self.max_dist+1),weights=corrected, bins=n_bins, density=False, alpha=0.6)
            linres = stats.linregress(range(self.min_dist,self.max_dist+1),np.array(corrected,dtype = float))
            plt.plot(x, linres.intercept + linres.slope*x, 'r', label='fitted line')
            plt.savefig(f'{save}corrected_{tf1}_{tf2}.png', dpi=600)
            plt.clf()
        
        return corrected
    
    def correct_all(self,n_bins = None, save = None):
        self.logger.info(f"Correcting background")
        corrected = {}
        if self.linres is None:
            self.logger.error("Please fit a linear regression first. [.linregress_all()]")
            sys.exit(0)
        for idx,row in self.linres.iterrows():
            tf1,tf2,linres = row
            res=self.correct_pair((tf1,tf2),linres,n_bins,save)
            corrected[tf1,tf2]=[tf1,tf2]+res
        
        self.corrected = pd.DataFrame.from_dict(corrected,orient="index",columns=['TF1','TF2']+[str(x) for x in range (self.min_dist, self.max_dist+1)]).reset_index(drop=True) 
        
    def get_median(self,tf1,tf2):
        if self.distances is None:
            self.logger.error("Can not calculate Median, no distances evaluated yet. Please run .count_distances() first.")
            sys.exit(0)
        data = self.distances.loc[((self.distances["TF1"]==tf1) &
               (self.distances["TF2"]==tf2))].iloc[0, 2:]

        self.logger.debug(f" Median for pair {tf1} - {tf2}: {data.median}")
        return data.median()

    def analyze_signal_pair(self, pair, corrected, smooth_window = 3, height = 0, prominence = None, save = None, new_file = True):
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
        """ Wrapper for analyze_signal_pair(). Will run the analysis for all pairs and saves results in the object itself. 
            
            Parameters
            ----------
            motif : str 
                Name of motif to select
                
            Returns
            -------
            new object with reduced rules and TFBS sets

        """
        self.logger.info(f"Analyzing Signal")
        all_peaks = []
        if smooth_window > 1:
            self.smooth(smooth_window)
        if self.corrected is None:
            self.logger.error("Background is not corrected yet. Please try .correct_all() first.")
            sys.exit(0)
        #TODO check save
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
                prominence = self.get_median(tf1,tf2)

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
        if (self.smoothed is None) or (self.smooth_window <= 1): 
            return False
        return True
        

    def check_periodicity(self):
        """ checks periodicity of distances (like 10 bp indicating DNA full turn)
            - placeholder for functionality upgrade -
            Returns:
		    ----------
            pd.DataFrame 
        """
        pass
    
    # TODO: move to objects.py
    def select_motif(self,motif):
        """ Select all motif related rules and TFBS names as new object. 
            
            Parameters
            ----------
            motif : str 
                Name of motif to select
                
            Returns
            -------
            new object with reduced rules and TFBS sets

        """
        self.logger.debug(f"Selecting Rules for motif {motif}. Don't forget to reestimate .count_distances()!")
        selected = self.rules.copy()
        selected = selected[(selected["TF1"]==motif) | (selected["TF2"]==motif)]
        new_obj = self.copy()
        new_obj.rules = selected

        selected_names = list(set(selected["TF1"].tolist() + selected["TF2"].tolist()))
        new_obj.TFBS = RegionList([site for site in self.TFBS if site.name in selected_names])

        return(new_obj)

    # TODO: remove code duplication  
    def get_pair_locations(self, TF1, TF2,TF1_strand = None, TF2_strand = None,min_distance = 0, max_distance = 100, max_overlap = 0,directional = False):
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

        ### (TODO: Check that .TFBS is filled) obsolete if duplication removed

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

										#Save location
                                        reg1 = OneRegion([TF1_chr, TF1_start, TF1_end, TF1_name, TF1_strand_i])
                                        reg2 = OneRegion([TF2_chr, TF2_start, TF2_end, TF2_name, TF2_strand_i])
                                        location = (reg1, reg2, distance)
                                        locations.append(location)

                            else:
								#The next site is out of window range; increment to next i
                                i += 1
                                finding_assoc = False   #break out of finding_assoc-loop
			
            else: #current TF1 is not TF1/TF2; go to next site
                i += 1

        return(locations)

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

    # TODO: duplicated
    def copy(self):
        """ Returns a copy of the DistObj """

        copied = copy.deepcopy(self)
        #copied.logger = TFcombLogger(self.verbosity) #receives its own logger
        return(copied)

	#-------------------------------------------------------------------------------------------------------------#
	#---------------------------------------------- plotting -----------------------------------------------------#
	#-------------------------------------------------------------------------------------------------------------#
    def plot_bar(self,targets,dataSource,save = None):
        """ Barplots for a list of TF-pairs

         Parameters
            ----------
            targets : array 
                Pairs (tuples) to create plots for.
            dataSource : pd.DataFrame 
                Source Data (should be a result Table from .get_raw_distances())
            save:
                Output Path to write results to. (filename will be constructed automatically from TF1-/TF2-name)
                Default: None (results will not be saved)

        """
        source_table = pd.DataFrame(dataSource)
        
        for pair in targets:
            fig = plt.bar(x = range(self.min_dist,self.max_dist+1), height=source_table.loc[((source_table["TF1"]==pair[0]) &
                                       (source_table["TF2"]==pair[1]))].iloc[0, 2:])
            #plt.xlim([0, 100])
            plt.title(pair)
            # TODO: Check if save is a valid path
            if save is not None:
                plt.savefig(f'{save}bar_{pair[0]}_{pair[1]}.png', dpi=600)
                plt.clf()

    def plot_hist(self,targets,dataSource,nbins=None,save = None):
        """ Histograms for a list of TF-pairs

         Parameters
            ----------
            targets : array 
                Pairs (tuples) to create plots for.
            dataSource : pd.DataFrame 
                Source Data (should be a result Table from .get_raw_distances())
            nbins: int
                Number of bins. Default: None (Binning is done automatically)
            save:
                Output Path to write results to. (filename will be constructed automatically from TF1-/TF2-name)
                Default: None (results will not be saved)

        """
        source_table = pd.DataFrame(dataSource)    
        for pair in targets:
            plt.hist(range(self.min_dist,self.max_dist+1), nbins, weights=source_table.loc[((source_table["TF1"]==pair[0]) &
                                       (source_table["TF2"]==pair[1]))].iloc[0, 2:])
            #plt.xlim([0, 100])
            plt.title(pair)
            # TODO: Check if save is a valid path
            if save is not None:
                plt.savefig(f'{save}hist_{pair[0]}_{pair[1]}.png', dpi=600)
                plt.clf()

    def plot_dens(self,targets,dataSource,bwadjust = 1,save = None):
        """ KDE Plots for a list of TF-pairs

            Parameters
            ----------
            targets : array 
                Pairs (tuples) to create plots for.
            dataSource : pd.DataFrame 
                Source Data (should be a result Table from .get_raw_distances())
            bwadjust: int
                Factor that multiplicatively scales the value chosen using bw_method. Increasing will make the curve smoother. 
                See kdeplot() from seaborn. Default: 1
            save:
                Output Path to write results to. (filename will be constructed automatically from TF1-/TF2-name)
                Default: None (results will not be saved)

        """
        source_table = pd.DataFrame(dataSource)
        
        for pair in targets:
            weights = list(source_table.loc[((source_table["TF1"]==pair[0]) &
                                       (source_table["TF2"]==pair[1]))].iloc[0, 2:])
            sns.kdeplot(range(self.min_dist,self.max_dist+1),weights = weights,bw_adjust=bwadjust,x="distance").set_title(pair)
            # TODO: Check if save is a valid path
            if save is not None:
                plt.savefig(f'{save}dens_{pair[0]}_{pair[1]}.png', dpi=600)
                plt.clf()

    def plot_decision_boundary(self,targets,n_bins = None, save = None):
        if n_bins is None:
            n_bins = self.max_dist - self.min_dist+1

        if self.corrected is None:
            self.logger.error("Background is not yet corrected. Please try .correct_all() first.")
            sys.exit(0)

        for pair in targets:
            tf1 = pair[0]
            tf2 = pair[1]
            corrected_data = self.corrected.loc[((self.corrected["TF1"]==tf1) &
                                           (self.corrected["TF2"]==tf2))].iloc[0, 2:]
            linres = stats.linregress(range(self.min_dist,self.max_dist+1),np.array(corrected_data,dtype = float))
            x = np.linspace(self.min_dist,self.max_dist+1, n_bins)
            thresh = self.get_median(tf1,tf2) * self.foldchange_thresh
            plt.hist(range(self.min_dist,self.max_dist+1),weights=corrected_data, bins=n_bins, density=False, alpha=0.6)
            plt.plot(x, [thresh]*len(x), 'r', label='upper boundary')
            plt.plot(x, [-thresh]*len(x), 'r', label='lower boundary')
            title = f"Decision boundary for {tf1}-{tf2}" 
            plt.title(title)
            if save is not None:
                plt.savefig(f'{save}db_{tf1}_{tf2}.png', dpi=600)
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
