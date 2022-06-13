from __future__ import print_function

import pandas as pd
import numpy as np
from scipy.stats import chisquare
import multiprocessing as mp
import seaborn as sns
import matplotlib.pyplot as plt

#Internal functions
from tfcomb.logging import TFcombLogger, InputError
from tfcomb.utils import check_columns, check_type, is_symmetric

#-------------------------------------------------------------------------------#
#---------------------------- Orientation analysis -----------------------------#
#-------------------------------------------------------------------------------#

def _get_scenario_keys(TF1, TF2, scenario):
			
		scenario_keys = {
						"same" :   [(TF1, "+", TF2, "+"),
									(TF2, "+", TF1, "+"),
									(TF1, "-", TF2, "-"),
									(TF2, "-", TF1, "-")],
						"opposite": [(TF1, "+", TF2, "-"),
									(TF1, "-", TF2, "+"),
									(TF2, "+", TF1, "-"),
									(TF2, "-", TF1, "+")],
						"TF1-TF2": [(TF1, "+", TF2, "+"), 
									(TF2, "-", TF1, "-")],
						"TF2-TF1": [(TF2, "+", TF1, "+"), 
									(TF1, "-", TF2, "-")],
						"convergent": [(TF1, "+", TF2, "-"), 
									   (TF2, "+", TF1, "-")],
						"divergent": [(TF1, "-", TF2, "+"), 
									  (TF2, "-", TF1, "+")]
						}

		return(scenario_keys[scenario])

def _get_unique_pairs(pairs):

	#Remove TF2-TF1 duplicates
	seen = {}
	for pair in pairs:
		if pair[::-1] not in seen: #the reverse TF2-TF1 pair has not been seen yet
			seen[pair] = ""
	unique = list(seen.keys())

	return(unique)

def orientation(rules, verbosity=1):
	"""
	Perform orientation analysis on the TF pairs in a directional / strand-specific table. The analysis counts different scenarios depending on the input.
	

	If the input matrix is symmetric, the analysis contains two scenarios::

		1. Same:      ⟝(TF1+)⟶  ⟝(TF2+)⟶  =  ⟝(TF2+)⟶  ⟝(TF1+)⟶  =  ⟵(TF1-)⟞  ⟵(TF2-)⟞  =  ⟵(TF2-)⟞  ⟵(TF1-)⟞
		2. Opposite:  ⟝(TF1+)⟶  ⟵(TF2-)⟞  =  ⟝(TF2+)⟶  ⟵(TF1-)⟞  =  ⟵(TF1-)⟞  ⟝(TF2+)⟶  =  ⟵(TF2-)⟞  ⟝(TF1+)⟶

	If the input is directional, the analysis contains four different scenarios::

		1. TF1-TF2:     ⟝(TF1+)⟶  ⟝(TF2+)⟶   =   ⟵(TF2-)⟞  ⟵(TF1-)⟞
		2. TF2-TF1:     ⟝(TF2+)⟶  ⟝(TF1+)⟶   =   ⟵(TF1-)⟞  ⟵(TF2-)⟞
		3. convergent:  ⟝(TF1+)⟶  ⟵(TF2-)⟞   =   ⟝(TF2+)⟶  ⟵(TF1-)⟞
		4. divergent:   ⟵(TF1-)⟞  ⟝(TF2+)⟶   =   ⟵(TF2-)⟞  ⟝(TF1+)⟶

	Parameters
	----------
	rules : pd.DataFrame
		The .rules output of a CombObj analysis.
	verbosity : int
		A value between 0-3 where 0 (only errors), 1 (info), 2 (debug), 3 (spam debug). Default: 1.

	Returns
	--------
	An OrientationAnalysis object (subclass of pd.DataFrame). The table contains frequencies of pairs related to each scenario.

	The dataframe has the following columns:
		- TF1: name of the first TF in pair
		- TF2: name of the second TF in pair
		- TF1_TF2_count: The total count of TF1-TF2 co-occurring pairs
		- If symmetric:
			- Same
			- Opposite
		- If directional:
			- TF1_TF2
			- TF2_TF1
			- convergent
			- divergent
		- std: Standard deviation of scenario frequencies
		- pvalue: A chi-square test to test the hypothesis that the scenarios are equally distributed

	"""
	logger = TFcombLogger(verbosity)

	#Check that input is dataframe
	check_type(rules, pd.DataFrame, "rules")
	rules = rules.copy() #ensures that rules-table is not changed
		
	#Split TF names from strands
	try:
		rules[["TF1_name", "TF1_strand"]] = rules["TF1"].str.split("(", expand=True)
	except Exception as e:
		raise InputError("Failed to split TF name from strand. Please ensure that .count_within() was run with '--stranded=True' and/or '--directional=True'.")
	rules["TF1_strand"] = rules["TF1_strand"].str.replace(")", "", regex=False)

	rules[["TF2_name", "TF2_strand"]] = rules["TF2"].str.split("(", expand=True)
	rules["TF2_strand"] = rules["TF2_strand"].str.replace(")", "", regex=False)

	#Establish if rules are directional
	rules_pivot = rules.pivot(index='TF1', columns='TF2', values='TF1_TF2_count')
	symmetric_bool = is_symmetric(rules_pivot)
	if symmetric_bool == True: 
		scenarios = ["same", "opposite"] #2 scenarios
		logger.info("Rules are symmetric - scenarios counted are: {0}".format(scenarios))

		#Remove duplicated pairs from analysis (TF1-TF2 = TF2-TF1)
		rules.set_index(["TF1", "TF2"], inplace=True)
		pairs = rules.index.tolist()
		unique = _get_unique_pairs(pairs)
		rules = rules.loc[unique,:]
	
	else:
		scenarios = ["TF1-TF2", "TF2-TF1", "convergent", "divergent"] #4 scenarios
		logger.info("Rules are directional - scenarios counted are: {0}".format(scenarios))

	#Setup count dictionary
	keys = [tuple(x) for x in rules[["TF1_name", "TF1_strand", "TF2_name", "TF2_strand"]].values]
	counts = rules["TF1_TF2_count"].tolist()
	count_dict = dict(zip(keys, counts))
	
	#Get all possible TF1-TF2 pairs
	pairs = list(zip(rules["TF1_name"], rules["TF2_name"]))
	pairs = sorted(list(set(pairs)))
	pairs = _get_unique_pairs(pairs)

	#Get counts per scenario
	counts = {}
	for (TF1, TF2) in pairs:
		
		counts[(TF1, TF2)] = {} #initialize counts per pair

		for scenario in scenarios:
			keys = _get_scenario_keys(TF1, TF2, scenario)
			keys = set(keys) #remove duplicate keys in case of TF1==TF2
			count = np.sum([count_dict.get(key, 0) for key in keys]) #sum of counts

			counts[(TF1, TF2)][scenario] = count

	#Create dataframe
	frame = pd.DataFrame().from_dict(counts, orient="index")
	frame.index.names = ["TF1", "TF2"]
	frame.reset_index(inplace=True)
	frame = frame[frame[scenarios].sum(axis=1) > 0] #remove any scenarios with sum of 0

	#Calculate chisquare p-value (are the scenarios normally distributed?)
	unique = frame[scenarios].drop_duplicates() #only calculate chi-square once per seen count combination - speeds up calculation
	mat = unique.to_numpy()
	rows, cols = mat.shape
	pvalues = [0]*rows
	for row in range(rows):
		n = mat[row,:] #counts per scenario
		s, p = chisquare(n)
		pvalues[row] = p
	unique["pvalue"] = pvalues
		
	#Merge unique to frame
	frame = frame.merge(unique, left_on=scenarios, right_on=scenarios, how="left")
	frame.index = frame["TF1"] + "-" + frame["TF2"]

	#Normalize counts to sum of 1
	frame["TF1_TF2_count"] = frame[scenarios].sum(axis=1)
	for scenario in scenarios:
		frame[scenario] = frame[scenario] / frame["TF1_TF2_count"]
		frame[scenario] = frame[scenario].replace(np.inf, 0)

	#Calculate standard deviation
	frame["std"] = frame[scenarios].std(axis=1)

	#Sort by pvalue/count and reorder colums in frame
	frame.sort_values(["pvalue", "TF1_TF2_count"], ascending=[True, False], inplace=True)
	frame = frame[["TF1", "TF2", "TF1_TF2_count"] + scenarios + ["std", "pvalue"]]
	frame = OrientationAnalysis(frame)

	return(frame)


class OrientationAnalysis(pd.DataFrame):
	""" Analysis of the orientation of TF co-ocurring pairs """
	
	@property
	def _constructor(self):
		return OrientationAnalysis

	def plot_heatmap(self, yticklabels=False, figsize=(6,6), save=None, **kwargs):
		""" Plot a heatmap of orientation scenarios for the output of the orientation analysis.
		
		Parameters
		-----------
		yticklabels : bool, optional
			Show yticklabels (TF-pairs) in plot. Default: False.
		figsize : tuple
			The size of the output heatmap. Default: (6,6)
		save : str, optional
			Save the plot to the file given in 'save'. Default: None.		 
		kwargs : arguments
			Any additional arguments are passed to sns.clustermap.

		Returns
		--------
		seaborn.matrix.ClusterGrid
		"""

		scenarios = [col for col in self.columns if col not in ["TF1", "TF2", "TF1_TF2_count", "std", "pvalue"]]

		g = sns.clustermap(self[scenarios],
							col_cluster=False,
							yticklabels=yticklabels,
							cbar_kws={'label': "Fraction"},
							figsize=figsize, 
							**kwargs)

		n_pairs = self.shape[0]
		g.ax_heatmap.set_ylabel("TF-TF pairs (n={0})".format(n_pairs))

		if save is not None:
			plt.savefig(save, dpi=600)

		return(g)
