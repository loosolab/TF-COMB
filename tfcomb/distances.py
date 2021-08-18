from tfcomb.logging import *
from tfcomb.counting import count_distances
from tobias.utils.regions import OneRegion, RegionList
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv 

#TODO: Include logger

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
        self.raw_distances = None 	 # Numpy array of size n_pairs x maxDist
        self.preferred = None 	     # Numpy array of size n_pairs x n_preferredDistance 
        self.n_bp = 0			     # Predicted number of baskets 
        self.TFBS = RegionList()     # None RegionList() of TFBS
        self.anchor_mode = 0         # Distance measure mode [0,1,2]
        self.name_to_idx = None      # Mapping TF-names: string <-> int 
        self.min_dist = 0            # Minimum distance. Default: 0 
        self.max_dist = 100          # Maximum distance. Default: 100
        self.max_overlap = 0         # Maximum overlap. Default: 0            

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
		
    def count_distances(self):
        """ Count distances for co_occurring TFs, can be followed by analyze_distances
            to determine preferred binding distances
        
        """
        chromosomes = {site.chrom:"" for site in self.TFBS}.keys()
        chrom_to_idx = {chrom: idx for idx, chrom in enumerate(chromosomes)}
        self.name_to_idx = {name: idx for idx, name in enumerate(self.TF_names)}
        sites = np.array([(chrom_to_idx[site.chrom], site.start, site.end, self.name_to_idx[site.name]) 
                          for site in self.TFBS]) #numpy integer array
	
        pairs = np.array([(self.name_to_idx[rule[0]], self.name_to_idx[rule[1]]) for rule in self.rules[["TF1","TF2"]].to_numpy()])
        self.raw_distances = count_distances(sites, 
                                             pairs,
                                             self.min_dist,
                                             self.max_dist,
                                             self.anchor_mode)
    
    def get_raw_distances(self):
        """ Get the raw distance in human readable format
            
            Returns:
		    ----------
			pd.Dataframe (TF1 name, TF2 name, count min_dist, count min_dist +1, ...., count max_dist)
        """
        results = []
        for index, row in self.rules.iterrows():
            tf1 = row["TF1"]
            tf2 = row["TF2"]
            entry =  [tf1,tf2]
            entry += self.raw_distances[index][2:].tolist()
            results.append(entry)
                
        return pd.DataFrame(results,columns=['TF1','TF2']+[str(x) for x in range (self.min_dist, self.max_dist+1)])

    def analyze_distances():
        """ Analyze preferred binding distances, requires count_distances() run.

            Returns:
		    ----------
            pd.DataFrame (all pairs found periodic)
        """
        # Statistical test / grid points? 
        pass

    def check_periodicity(self):
        """ checks periodicity of distances (like 10 bp indicating dna full turn)

            Returns:
		    ----------
            pd.DataFrame (all pairs found periodic)
        """
        pass
    
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
                        # TODO: Check dist_range
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


