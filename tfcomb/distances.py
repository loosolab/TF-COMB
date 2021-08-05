from tfcomb.logging import *
from tobias.utils.regions import OneRegion, RegionList


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
        self.preferred = None 	     #numpy array of size n_TFs x n_TFs x maxDist
        self.n_bp = 0			     #predict the number of baskets 
        self.TFBS = RegionList()     #None RegionList() of TFBS
        self.anchor = 0              #distance measure mode [0,1,2]

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


    def set_anchor(anchor):
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
        self.anchor = modes.index(anchor)
		
    def count_distances():
        """ count distances for co_occurring TFs, can be followed by analyze_distances
            to determine preferred binding distances
        
        """
        # call count method from counting.pyx
        # fill raw_distances (key will be (tf1,tf2,dist), value = count)

        pass
    
    def analyze_distances():
        """ Analyze preferred binding distances, requires count_distances() run.

        """
        # Statistical test?
        pass

    def check_periodicity():
        """ checks periodicity of distances (like 10 bp indicating dna full turn)

        return: DataFrame (all pairs found periodic)
        """
        pass

    def plot_histogram():
        pass

    def plot_dens():
        pass

