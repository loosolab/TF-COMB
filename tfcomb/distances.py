from sys import path, stderr
from tfcomb.logging import *
from tfcomb.counting import count_distances
from tobias.utils.regions import OneRegion, RegionList
from tobias.utils.signals import fast_rolling_math
from scipy.signal import find_peaks
from tfcomb.logging import *
import tfcomb.utils
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
        self._raw = None             # Raw distance data [Numpy array of size n_pairs x maxDist]

        self.distances = None 	     # Pandas DataFrame of size n_pairs x maxDist
        self.corrected = None        # Pandas DataFrame of size n_pairs x maxDist
        self.linres = None           # Pandas DataFrame of size n_pairs x maxDist
        self.normalized = None       # Pandas DataFrame of size n_pairs x maxDist
        self.smoothed = None         # Pandas DataFrame of size n_pairs x maxDist
        self.peaks = None 	         # Pandas DataFrame of size n_pairs x n_preferredDistance 

        self.peaking_count = None    # Number of pairs with at least one peak 
        self.directional = None      # True if direction is taken into account, false otherwise 
        
        self.smooth_window = 3       # Smoothing window size, 1 = no smoothing
        self.n_bp = 0			     # Predicted number of baskets 
        self.TFBS = RegionList()     # None RegionList() of TFBS
        self.anchor_mode = 0         # Distance measure mode [0,1,2]
        self.name_to_idx = None      # Mapping TF-names: string <-> int 
        self.pair_to_idx = None      # Mapping Pairs: tuple(string) <-> int
        self.min_dist = 0            # Minimum distance. Default: 0 
        self.max_dist = 300          # Maximum distance. Default: 100
        self.max_overlap = 0         # Maximum overlap. Default: 0       
   
        self._PEAK_HEADER = "TF1\tTF2\tDistance\tPeak Heights\tProminences\tProminence Threshold\n"

    def __str__(self):
	    pass
    
    def set_verbosity(self, level):
	    """ Set the verbosity level for logging after creating the CombObj.

		Parameters
		----------
		level : int
			A value between 0-3 where 0 (only errors), 1 (info), 2 (debug), 3 (spam debug). 
        
        Returns
		----------
		None 
			Sets the verbosity level for the Logger inplace
		"""

	    self.verbosity = level
	    self.logger = TFcombLogger(self.verbosity) #restart logger with new verbosity	    
    
    def fill_rules(self,comb_obj):
        """ Fill DistanceObject according to reference object with all needed Values and parameters
        to perform standard prefered distance analysis

        Parameters
		----------
		comb_obj: tfcomb.objects
            Object from which the rules and parameters should be copied from

        Returns
		----------
		None 
			Copies values and parameters from a combObj or diffCombObj.
        
        try:
            tfcomb.utils.check_type(comb_obj,[CombObj,DiffCombObj],"CombObject")
        except ValueError as e :
            self.logger.error(str(e))
            sys.exit(0)
        """
        
        #copy rules
        self.rules = comb_obj.rules
        # reset pandas index
        self.rules = self.rules.reset_index(drop=True)

        # copy parameters
        self.TF_names = comb_obj.TF_names
        self.TFBS = comb_obj.TFBS 
        self.min_dist = comb_obj.min_distance
        self.max_dist = comb_obj.max_distance
        self.directional = comb_obj.directional
        self.max_overlap = comb_obj.max_overlap
        self.anchor = comb_obj.anchor

    def set_anchor(self,anchor):
        """ set anchor for distance measure mode
        0 = inner
        1 = outer
        2 = center

        Parameters
		----------
		anchor : str or int
            one of ["inner","outer","center"] or [0,1,2]

        Returns
		----------
		None 
			Sets anchor mode inplace
        """

        try:
            tfcomb.utils.check_type(anchor,[str,int],"anchor")
        except ValueError as e:
            self.logger.error(str(e))
            sys.exit(0)

        if isinstance(anchor,str):
            try:
                tfcomb.utils.check_string(anchor,["inner","outer","center"])
            except ValueError as e:
                self.logger.error(str(e))
                sys.exit(0)

            modes = ["inner","outer","center"]
            self.anchor_mode = modes.index(anchor)
        # anchor is int
        else:
            try:
                tfcomb.utils.check_value(anchor,0,2)
            except ValueError as e:
                self.logger.error(str(e))
                sys.exit(0)
            self.anchor_mode = anchor
		
    def count_distances(self, normalize = True, directional = False):
        """ Count distances for co_occurring TFs, can be followed by analyze_distances
            to determine preferred binding distances

        Parameters
		----------
		normalize : bool
            True if data should be normalized, False otherwise. Normalization is done as followed:
            (number of counted occurrences for a given pair at a given distance) / (Total amount of occurrences for the given pair)
            Default: True
        directional : bool
			Decide if direction of found pairs should be taken into account, e.g. whether  "<---TF1---> <---TF2--->" is only counted as 
			TF1-TF2 (directional=True) or also as TF2-TF1 (directional=False). Default: False.
        
        Returns
		----------
		None 
			Fills the object variable .distances.

        """
        chromosomes = {site.chrom:"" for site in self.TFBS}.keys()
        # encode chromosome,pairs and name to int representation
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
        
        # Unify (directional) counts 
        if not directional:
            for i in range(0,self._raw.shape[0]-1):
                if (self._raw[i,0] == self._raw[i+1,1]) and (self._raw[i,1] == self._raw[i+1,0]):
                    s = self._raw[i,2:]+self._raw[i+1,2:]
                    self._raw[i,2:] = s
                    self._raw[i+1,2:] = s
        self.directional = directional

        # convert raw counts (numpy array with int encoded pair names) to better readable format (pandas DataFrame with TF names)
        self._raw_to_human_readable(normalize)

        self.logger.info("Done finding distances! Run .linregress_pair() or .linregress_all() to fit linear regression")
    
    def _raw_to_human_readable(self, normalize = True):
        """ Get the raw distance in human readable format
            
            Parameters
		    ----------
            normalize : bool
            True if data should be normalized, False otherwise. Normalization is done as followed:
            (number of counted occurrences for a given pair at a given distance) / (Total amount of occurrences for the given pair)
            Default: True

            Returns:
		    ----------
			pd.Dataframe (TF1 name, TF2 name, count min_dist, count min_dist +1, ...., count max_dist)
        """
        self.logger.debug("Converting raw count data to pretty dataframe")
        idx_to_name = {}
        # get names from int encoding
        for k,v in self.name_to_idx.items():
            idx_to_name[v] = k 
        
        results = []
        for row in self._raw:
            tf1 = idx_to_name[row[0]]
            tf2 = idx_to_name[row[1]]
            entry = [tf1,tf2]
            
            if normalize:
                entry += (row[2:]/(row[2:].sum())).tolist()
            else:
                entry += row[2:].tolist()
            results.append(entry)

        self.normalized = normalize    
        self.distances = pd.DataFrame(results,columns=['TF1','TF2']+[str(x) for x in range (self.min_dist, self.max_dist+1)])

    def linregress_pair(self,pair,n_bins=None, save = None):
        """ Fits a linear Regression to distance count data for a given pair. The linear regression is used to 
            estimate the background. Proceed with .correct_pair()
            
            Parameters
		    ----------
            pair : tuple(str,str)
                TF names for which the linear regression should be performed. e.g. ("NFYA","NFYB")
            n_bins: int 
                Number of bins used for plotting. If n_bins is none, binning resolution is one bin per data point. 
                Default: None
            save: str
                Path to save the plots to. If save is None plots won't be plotted. 
                Default: None

            Returns:
		    ----------
			scipy.stats._stats_mstats_common.LinregressResult Object
        """
        try:
            tfcomb.utils.check_type(pair,[tuple],"pair")
        except ValueError as e:
            self.logger.error(str(e))
            sys.exit(0)
        
        try:
            tfcomb.utils.check_type(n_bins,[int,type(None)],"n_bins")
        except ValueError as e:
            self.logger.error(str(e))
            sys.exit(0)
        
        try:
            tfcomb.utils.check_writeability(save)
        except ValueError as e:
            self.logger.error(str(e))
            sys.exit(0)

        if self.distances is None:
            self.logger.error("No distances evaluated yet. Please run .count_distances() first.")
            sys.exit(0)
        #TODO: check pair is valid
        tf1 = pair[0]
        tf2 = pair[1]

        self.logger.debug(f"Fitting linear regression for pair: {pair}")
        if n_bins is None:
            n_bins = self.max_dist - self.min_dist +1
        x = np.linspace(self.min_dist,self.max_dist+1, n_bins)
        
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
        """ Fits a linear Regression to distance count data for all rules. The linear regression is used to 
            estimate the background. Proceed with .correct_all()
            
            Parameters
		    ----------
            n_bins: int 
                Number of bins used for plotting. If n_bins is none, binning resolution is one bin per data point. 
                Default: None
            save: str 
                Path to save the plots to. If save is None plots won't be plotted. 
                Default: None

            Returns:
		    ----------
			None
                Fills the object variable .linres
        """

        try:
            tfcomb.utils.check_type(n_bins,[int,type(None)],"n_bins")
        except ValueError as e:
            self.logger.error(str(e))
            sys.exit(0)

        try:
            tfcomb.utils.check_writeability(save)
        except ValueError as e:
            self.logger.error(str(e))
            sys.exit(0)

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
        """ Subtracts the estimated background from the Signal for a given pair. 
            
            Parameters
		    ----------
            pair : tuple(str,str)
                TF names for which the background correction should be performed. e.g. ("NFYA","NFYB")
            linres: scipy.stats._stats_mstats_common.LinregressResult 
                Fitted linear regression for the given pair
            n_bins: int 
                Number of bins used for plotting. If n_bins is none, binning resolution is one bin per data point. 
                Default: None
            save: str
                Path to save the plots to. If save is None plots won't be plotted. 
                Default: None

            Returns:
		    ----------
			list 
                Corrected values for the given pair
        """
        try:
            tfcomb.utils.check_type(pair,[tuple],"pair")
        except ValueError as e:
            self.logger.error(str(e))
            sys.exit(0)

        try:
            tfcomb.utils.check_type(n_bins,[int,type(None)],"n_bins")
        except ValueError as e:
            self.logger.error(str(e))
            sys.exit(0)

        try:
            tfcomb.utils.check_writeability(save)
        except ValueError as e:
            self.logger.error(str(e))
            sys.exit(0)

        #TODO: check pair is valid & check linres: import in utils needed ?
        tf1 = pair[0]
        tf2 = pair[1]

        if linres is None:
            self.logger.error("Please fit a linear regression first. [see .linregress_pair()]")
            sys.exit(0)

        self.logger.debug(f"Correcting background for pair {pair}")
        if n_bins is None:
            n_bins = self.max_dist - self.min_dist +1
       
        if self.distances is None:
            self.logger.error("No distances evaluated yet. Please run .count_distances() first.")
            sys.exit(0)
        data = self.distances.loc[((self.distances["TF1"]==tf1) &
               (self.distances["TF2"]==tf2))].iloc[0, 2:]
        corrected = []
        x_val = 0
        
        for dist in data:
            # subtract background from signal
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
        """ Subtracts the estimated background from the Signal for all rules. 
            
            Parameters
		    ----------
            pair : tuple(str,str)
                TF names for which the background correction should be performed. e.g. ("NFYA","NFYB")

            Returns:
		    ----------
			None 
                Fills the object variable .corrected
        """
        try:
            tfcomb.utils.check_type(n_bins,[int,type(None)],"n_bins")
        except ValueError as e:
            self.logger.error(str(e))
            sys.exit(0)

        try:
            tfcomb.utils.check_writeability(save)
        except ValueError as e:
            self.logger.error(str(e))
            sys.exit(0)
        
        if self.linres is None:
            self.logger.error("Please fit a linear regression first. [see .linregress_all()]")
            sys.exit(0)

        self.logger.info(f"Correcting background")
        corrected = {}
        
        for idx,row in self.linres.iterrows():
            tf1,tf2,linres = row
            res=self.correct_pair((tf1,tf2),linres,n_bins,save)
            corrected[tf1,tf2]=[tf1,tf2]+res
        
        self.corrected = pd.DataFrame.from_dict(corrected,orient="index",columns=['TF1','TF2']+[str(x) for x in range (self.min_dist, self.max_dist+1)]).reset_index(drop=True) 
        
    def get_median(self,pair):
        """ Estimates the median from the distinct counts per distance for a given pair.
            
            Parameters
		    ----------
            pair: tuple(str,str)
                TF names for which median should be calculated. e.g. ("NFYA","NFYB")
  
            Returns:
		    ----------
			Float 
                Median for the given pair 
        """
        try:
            tfcomb.utils.check_type(pair,[tuple],"pair")
        except ValueError as e:
            self.logger.error(str(e))
            sys.exit(0)

        if self.distances is None:
            self.logger.error("Can not calculate Median, no distances evaluated yet. Please run .count_distances() first.")
            sys.exit(0)
        
        # TODO: check pair is valid
        tf1 = pair[0]
        tf2 = pair[1]

        data = self.distances.loc[((self.distances["TF1"]==tf1) &
               (self.distances["TF2"]==tf2))].iloc[0, 2:]

        self.logger.debug(f" Median for pair {tf1} - {tf2}: {data.median}")
        return data.median()

    # TODO: Check if kwargs is better suited tham height & prominence
    def analyze_signal_pair(self, pair, corrected, smooth_window = 3, height = 0, prominence = 0, save = None, new_file = True):
        """ After background correction is done (see .correct_pair() or .correct_all()), the signal is analyzed for peaks, 
            indicating prefered binding distances. There can be more than one peak (more than one prefered binding distance) per 
            Signal. Peaks are called with scipy.signal.find_peaks().
            
            Parameters
		    ----------
            pair : tuple(str,str)
                TF names for which the background correction should be performed. e.g. ("NFYA","NFYB")
            corrected: list 
                corrected value for the given pair
            smooth_window: int 
                window size for the rolling smoothing window. A bigger window produces larger flanking ranks at the sides.
                (see tobias.utils.signals.fast_rolling_math) 
                Default: 3
            height: number or ndarray or sequence
                height parameter for peak calling (see scipy.signal.find_peaks() for detailed information). 
                Zero means only positive peaks are called.
                Default: 0
            prominence: number or ndarray or sequence
                prominence parameter for peak calling (see scipy.signal.find_peaks() for detailed information)
                Default: 0
            save: str
                Path to save the plots to. If save is None plots won't be plotted. 
                Default: None
            new_file: boolean
                True means results are written to a new file (overwrites already existing results), False means results are appended if 
                file already exists.
                Default: True

            Returns:
		    ----------
			list 
                list of found peaks in form [TF1, TF2, Distance, Peak Heights, Prominences, Prominence Threshold]
        """
        try:
            tfcomb.utils.check_type(pair,[tuple],"pair")
        except ValueError as e:
            self.logger.error(str(e))
            sys.exit(0)

        try:
            tfcomb.utils.check_type(smooth_window,[int],"smooth_window")
        except ValueError as e:
            self.logger.error(str(e))
            sys.exit(0)
        
        try:
            tfcomb.utils.check_type(corrected,[list],"corrected")
        except ValueError as e:
            self.logger.error(str(e))
            sys.exit(0)

        try:
            tfcomb.utils.check_writeability(save)
        except ValueError as e:
            self.logger.error(str(e))
            sys.exit(0)

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
        """ Helper function for smoothing all rules with a given window size. The function .correct_all() is required to be run beforehand.
            
            Parameters
		    ----------
            window_size: int 
                window size for the rolling smoothing window. A bigger window produces larger flanking ranks at the sides.
                (see tobias.utils.signals.fast_rolling_math) 
                Default: 3

            Returns:
		    ----------
			None 
                Fills the object variable .smoothed
        """
        try:
            tfcomb.utils.check_type(window_size,[int],"window size")
        except ValueError as e:
            self.logger.error(str(e))
            sys.exit(0)

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
        """ After background correction is done (see .correct_all()), the signal is analyzed for peaks, 
            indicating prefered binding distances. There can be more than one peak (more than one prefered binding distance) per 
            Signal. Peaks are called with scipy.signal.find_peaks().
            
            Parameters
		    ----------
            smooth_window: int 
                window size for the rolling smoothing window. A bigger window produces larger flanking ranks at the sides.
                (see tobias.utils.signals.fast_rolling_math) 
                Default: 3
            height: number or ndarray or sequence
                height parameter for peak calling (see scipy.signal.find_peaks() for detailed information). 
                Zero means only positive peaks are called.
                Default: 0
            prominence: number or ndarray or sequence or "median"
                prominence parameter for peak calling (see scipy.signal.find_peaks() for detailed information). 
                If "median", the median for the pairs is used (see .get_median())
                Default: "median"
            save: str
                Path to save the plots to. If save is None plots won't be plotted. 
                Default: None
            new_file: boolean
                True means results are written to a new file (overwrites already existing results), False means results are appended if 
                file already exists.
                Default: True

            Returns:
		    ----------
			None 
                Fills the object variable self.peaks, self.smooth_window, self.peaking_count
        """
        try:
            tfcomb.utils.check_type(smooth_window,[int],"smooth_window")
        except ValueError as e:
            self.logger.error(str(e))
            sys.exit(0)

        try:
            tfcomb.utils.check_writeability(save)
        except ValueError as e:
            self.logger.error(str(e))
            sys.exit(0)
        
        if smooth_window > 1:
            self.smooth(smooth_window)
        if self.corrected is None:
            self.logger.error("Background is not corrected yet. Please try .correct_all() first.")
            sys.exit(0)

        if isinstance(prominence,str):
            try:
                tfcomb.utils.check_string(prominence,["median"])
            except ValueError as e:
                self.logger.error(str(e))
                sys.exit(0)

        self.logger.info(f"Analyzing Signal")
        all_peaks = []

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
                prominence = self.get_median((tf1,tf2))

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
        """ Return True if data was smoothed during analysis, False otherwise
            
            Parameters
		    ----------
           
            Returns:
		    ----------
			bool 
                True if smoothed, False otherwiese
        """
        
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
        
        b = tfcomb.utils.get_pair_locations(TF1, TF2, TF1_strand = TF1_strand,
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
