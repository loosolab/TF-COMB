from __future__ import print_function

import os
import pandas as pd
import numpy as np
import itertools
import scipy
from scipy.stats import chisquare
import re
import copy
import multiprocessing as mp

#Internal functions
from tfcomb.logging import TFcombLogger, InputError
from tfcomb.utils import check_columns

import seaborn as sns

#-------------------------------------------------------------------------------#
#-------------------------- Directionality analysis ----------------------------#
#-------------------------------------------------------------------------------#

def _get_scenario_keys(TF1, TF2, scenario):
		
	scenario_keys = {
					 "scenario1": [(TF1, "+", TF2, "+"), 
								   (TF2, "-", TF1, "-")],
					 "scenario2": [(TF2, "+", TF1, "+"), 
								   (TF1, "-", TF2, "-")],
					 "scenario3": [(TF1, "+", TF2, "-"), 
								   (TF2, "+", TF1, "-")],
					 "scenario4": [(TF1, "-", TF2, "+"), 
								   (TF2, "-", TF1, "+")]
					}

	return(scenario_keys[scenario])

def directionality(rules):
	"""
	Perform directionality analysis on the TF pairs in a directional / strand-specific table.

	1. TF1-TF2:    |---TF1(+)--->   |---TF2(+)--->   =   <---TF2(-)---|   <---TF1(-)---| 
	2. TF2-TF1:    |---TF2(+)--->   |---TF1(+)--->   =   <---TF1(-)---|   <---TF2(-)---|
	3. convergent: |---TF1(+)--->   <---TF2(-)---|   =   |---TF2(+)--->   <---TF1(-)---| 
	4. divergent:  <---TF1(-)---|   |---TF2(+)--->   =   <---TF2(-)---|   |---TF1(+)--->

	Parameters
	----------
	rules : pd.DataFrame
		The .rules output of a CombObj analysis.

	Returns
	--------
	pd.DataFrame
		Percentages of pairs related to each scenario

		The dataframe has the following columns:
			- TF1: name of the first TF in pair
			- TF2: name of the second TF in pair
			- TF1_TF2_count: 
			- scenario1_TF1_TF2
			- scenario2_TF2_TF1
			- scenario3_convergent
			- scenario4_divergent
			- pvalue

	"""
	
	#TODO: Test input format
	tfcomb.utils.check_type(rules, pd.DataFrame, "rules")

	rules = rules.copy() #ensures that rules-table is not changed
	
	#Split TF names from strands
	try:
		rules[["TF1_name", "TF1_strand"]] = rules["TF1"].str.split("(", expand=True)
	except Exception as e:
		raise InputError("Failed to split TF name from strand. Please ensure that .count_within() was run with '--directionality=True' and '--stranded=True'. Exception was: {0}".format(e.message))
	rules["TF1_strand"] = rules["TF1_strand"].str.replace(")", "", regex=False)

	rules[["TF2_name", "TF2_strand"]] = rules["TF2"].str.split("(", expand=True)
	rules["TF2_strand"] = rules["TF2_strand"].str.replace(")", "", regex=False)
	
	#Setup count dictionary
	keys = [tuple(x) for x in rules[["TF1_name", "TF1_strand", "TF2_name", "TF2_strand"]].values]
	counts = rules["TF1_TF2_count"].tolist()
	count_dict = dict(zip(keys, counts))
	
	#Get all possible TF1-TF2 pairs
	pairs = list(zip(rules["TF1_name"], rules["TF2_name"]))
	pairs = list(set(pairs))

	#Remove TF2-TF1 duplicates
	seen = {}
	for pair in pairs:
		if pair[::-1] not in seen: #the reverse TF2-TF1 pair has not been seen yet
			seen[pair] = ""
	pairs = list(seen.keys())

	scenarios = ["scenario" + str(i) for i in range(1,5)]
	lines = []
	for (TF1, TF2) in pairs:
		
		counts = []
		for scenario in scenarios:
			keys = _get_scenario_keys(TF1, TF2, scenario)
			count = np.sum([count_dict.get(key, 0) for key in keys]) #sum of counts
			counts.append(count)
		
		#Normalize to sum of 1
		total_counts = np.sum(counts)

		## Collect results in table
		line = [TF1, TF2, total_counts] + counts
		lines.append(line)       

	columns = ["TF1", "TF2", "TF1_TF2_count"] + scenarios
	frame = pd.DataFrame(lines, columns=columns)
	
	#Calculate chisquare
	unique = frame[scenarios].drop_duplicates()
	mat = unique.to_numpy()
	rows, cols = mat.shape
	pvalues = [0]*rows
	for row in range(rows):
		n = mat[row,:]
		s, p = chisquare(n)
		pvalues[row] = p
	unique["pvalue"] = pvalues
	
	#Merge unique to frame
	frame = frame.merge(unique, left_on=scenarios, right_on=scenarios, how="left")

	#Normalize counts to sum of 1
	n = frame["TF1_TF2_count"].tolist()
	for scenario in scenarios:
		frame[scenario] = frame[scenario] / frame["TF1_TF2_count"]
		frame[scenario] = frame[scenario].replace(np.inf, 0)

	#Calculate standard deviation
	frame["std"] = frame[scenarios].std(axis=1)
	frame = frame[columns + ["std", "pvalue"]] #reorder columns

	#Sort by pvalue and number of co-occurrences found
	frame["s"] = -frame["TF1_TF2_count"]
	frame.sort_values(["pvalue", "s"], inplace=True)
	frame.drop(columns=["s"], inplace=True)

	#Rename scenarios
	frame.rename(columns={"scenario1": "scenario1_TF1-TF2",
				  "scenario2": "scenario2_TF2-TF1",
				  "scenario3": "scenario3_convergent",
				  "scenario4": "scenario4_divergent"}, inplace=True) 

	#Merge counts for same-TF pairs
	

	return(frame)

def plot_directionality(table):
	""" Plot directionality heatmap for the output of 'directionality' """

	scenario_columns = ["scenario1_TF1-TF2", "scenario2_TF2-TF1", "scenario3_convergent", "scenario4_divergent"]

	h = sns.clustermap(table[scenario_columns])

