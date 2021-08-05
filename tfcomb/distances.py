from tfcomb.logging import *
from tfcomb.counting import count_distances
from tobias.utils.regions import OneRegion, RegionList
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#TODO: Include logger

class DistObj():
    """
	The main class for analyzing preferred binding distances for co-occurring TFs.

	Examples
    ----------

	>>> D = tfcomb.distances.DistObj()

	#Verbosity of the output log can be set using the 'verbosity' parameter:
	>>> D = tfcomb.distances.DistObj(verbosity=2)

	""" 

    def __init__(self, verbosity = 1): #set verbosity 

		#Function and run parameters
        self.verbosity = verbosity  #0: error, 1:info, 2:debug, 3:spam-debug
        self.logger = TFcombLogger(self.verbosity)
        
        #Variables for storing data
        self.rules = None  		     #filled in by .fill_rules()
        self.TF_names = []		     #List of TF names
        self.raw_distances = None 	 #numpy array of size n_TFs x n_TFs x maxDist
        self.undirected_distances = None
        self.preferred = None 	     #numpy array of size n_TFs x n_TFs x 
        self.n_bp = 0			     #predict the number of baskets 
        self.TFBS = RegionList()     #None RegionList() of TFBS
        self.anchor_mode = 0              #distance measure mode [0,1,2]
        self.name_to_idx = None
        self.normalized = None

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
        self.TF_names = comb_obj.TF_names
        self.TFBS = comb_obj.TFBS 

        pass


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
        """ count distances for co_occurring TFs, can be followed by analyze_distances
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
                                             -100,
                                             100,
                                             self.anchor_mode)
    
    def get_raw_distances(self):
        r = 100
        results = []
        for index, row in self.rules.iterrows():
            tf1 = row["TF1"]
            tf2 = row["TF2"]
            entry =  [tf1,tf2]
            entry += self.raw_distances[self.name_to_idx[tf1]][self.name_to_idx[tf2]][:].tolist()
            results.append(entry)
                
        return pd.DataFrame(results,columns=['TF1','TF2']+[str(x) for x in range (-r,r+1)])

    def count_undirected(self):
        """
        """
        
        result = []
        n_rows = len(self.raw_distances)
        for ind in range(0,n_rows-1):
            row_cur = self.raw_distances[ind]
            row_next = self.raw_distances[ind+1]

            if (row_cur[0] == row_next[1] and row_cur[1] == row_next[0]):
                merged = [row_cur[0], row_cur[1]]
                mergedValues = row_cur + row_next
                [merged.append(x) for x in mergedValues[2:]]
                result.append(merged)
        
        self.undirected_distances = result

    
    def analyze_distances():
        """ Analyze preferred binding distances, requires count_distances() run.

        """
        # Statistical test?
        pass

    def check_periodicity(self):
        """ checks periodicity of distances (like 10 bp indicating dna full turn)

        return: DataFrame (all pairs found periodic)
        """
        pass

    def remove_background(self):
        #TODO: implement real method
        dists = self.get_raw_distances()
        summed = []
        for i in range(-100,101):
            summed.append(dists[f'{i}'].sum())
        norm = dists.iloc[:,2:] / summed
        names = dists.iloc[:,0:2]    

        self.normalized = pd.concat([names, norm], axis=1)


    def plot_bar(self,targets,dataSource,save = None):
        source_table = pd.DataFrame(dataSource)
        
        for pair in targets:
            fig = plt.bar(x = range(-100,101), height=source_table.loc[((source_table["TF1"]==pair[0]) &
                                       (source_table["TF2"]==pair[1]))].iloc[0, 2:])
            #plt.xlim([0, 100])
            plt.title(pair)
            if save is not None:
                plt.savefig(save, dpi=600)
                plt.clf()

    def plot_hist(self,targets,dataSource,nbins=201,save = None):
        source_table = pd.DataFrame(dataSource)    
        for pair in targets:
            plt.hist(range(-100,101), nbins, weights=source_table.loc[((source_table["TF1"]==pair[0]) &
                                       (source_table["TF2"]==pair[1]))].iloc[0, 2:])
            #plt.xlim([0, 100])
            plt.title(pair)
            if save is not None:
                plt.savefig(save, dpi=600)
                plt.clf()


    def plot_dens(self,targets,dataSource,bwadjust = 1,save = None):
        source_table = pd.DataFrame(dataSource)
        
        for pair in targets:
            weights = list(source_table.loc[((source_table["TF1"]==pair[0]) &
                                       (source_table["TF2"]==pair[1]))].iloc[0, 2:])
            sns.kdeplot(range(-100,101),weights = weights,bw_adjust=bwadjust,x="distance").set_title(pair)
            if save is not None:
                plt.savefig(save, dpi=600)
                plt.clf()


