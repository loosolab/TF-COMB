from __future__ import print_function

import os
import pkg_resources
import pandas as pd
import numpy as np
import itertools
import scipy
from tfcomb.utils import check_columns
from scipy.stats import chisquare

#Network analysis
import networkx as nx
import community as community_louvain

#GO-term analysis
from goatools.base import download_ncbi_associations
from goatools.base import download_go_basic_obo
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
from goatools.obo_parser import GODag

#UROPA annotation
import logging
import pysam
from uropa.annotation import annotate_single_peak
from uropa.utils import format_config

DATA_PATH = pkg_resources.resource_filename("tfcomb", 'data/')

name_to_taxid = {"human": 9606, 
				 "mouse": 10090}


def _fmt_field(v):
	""" Format a object to string depending on type. For use when formatting table entries of tuple/list """

	if isinstance(v, str):
		s = v
	elif isinstance(v, tuple):
		s = "/".join([str(i) for i in v])
	elif isinstance(v, list) or isinstance(v, set):
		o = sorted(list(v))
		s = ", ".join([str(i) for i in o])
	else: #float/int
		s = v
	return(s)

def go_enrichment(gene_ids, organism="human", background_gene_ids=None):
	"""
	Perform a GO-term enrichment based on the input gene_ids. 

	Parameters
	-----------
	gene_ids : list
		A list of gene ids
	organism : :obj:`str`, optional
		The organism of which the gene_ids originate. If organism 'background_gene_ids' are given, organism is not needed. Defaults to 'human'.
	background_gene_ids : list, optional
		A specific list of background gene ids to use. Default: uniprot proteins of the 'organism' given. 

	See also
	---------
	tfcomb.plotting.go_bubble

	Returns
	--------
	pd.DataFrame

	Reference
	----------

	"""
	
	#verbosity 0/1/2
	#setup logger
	
	#Gene_ids must be a list
	
	
	#Organism must be in human/mouse
	if organism not in name_to_taxid:
		raise ValueError("Organism '{0}' not available. Please choose any of: {1}".format(organism, list(name_to_taxid.keys())))
	else:
		taxid = name_to_taxid[organism]
	
	##### Read data #####
	#Setup GOATOOLS GO DAG
	obo_fname = download_go_basic_obo()
	obodag = GODag(obo_fname)
	
	#Setup gene -> GO term associations
	fin_gene2go = download_ncbi_associations()
	objanno = Gene2GoReader(fin_gene2go, taxids=[taxid])
	ns2assoc = objanno.get_ns2assc()
	
	#Read data from package
	gene_table = pd.read_csv(DATA_PATH + organism + "_genes.txt", sep="\t")
	
	##### Setup analysis ####
	
	#Setup background gene ids 
	if background_gene_ids == None:
		background_gene_ids = list(set(gene_table["GeneID"].tolist())) #unique gene ids from table
	else:
		pass
		#check if background_gene_ids are in ns2assoc
	
	#Setup goeaobj
	goeaobj = GOEnrichmentStudyNS(
				background_gene_ids, # List of protein-coding genes
				ns2assoc, # geneid/GO associations
				obodag, # Ontologies
				propagate_counts = False,
				alpha = 0.05, # default significance cut-off
				methods = ['fdr_bh']) # defult multipletest correction method
	
	##### Run study #####
	
	#Check if gene_ids are in ns2assoc; else, try to convert
	all_gene_ids = set(sum([list(ns2assoc[aspect].keys()) for aspect in ns2assoc.keys()], []))
	n_found = sum([gene_id in all_gene_ids for gene_id in gene_ids])
	
	if n_found == 0:
		
		#Which column from gene_table has the best match?
		match = []
		for column in gene_table.columns:
			tup = (column, sum([gene_id in gene_table[column].tolist() for gene_id in gene_ids]))
			match.append(tup)
		
		id_col = sorted(match, key=lambda t: -t[1])[0][0] #column with best matching IDs
		
		#Convert to entrez id
		ids2entrez = dict(zip(gene_table[id_col], gene_table["GeneID"]))
		gene_ids_entrez = [ids2entrez.get(s, -1) for s in gene_ids]
		gene_ids_entrez = set([gene for gene in gene_ids_entrez if gene > 0]) #only genes for which there was a match
	else:
		gene_ids_entrez = gene_ids
		
	goea_results_all = goeaobj.run_study(gene_ids_entrez)
	
	#Convert study items to original ids


	##### Format to dataframe #####

	keys_to_use = ["GO", "name", "NS", "depth", "enrichment", "ratio_in_study", "ratio_in_pop", 
				   "p_uncorrected", "p_fdr_bh", "study_count", "study_items"]
	lines = [[_fmt_field(res.__dict__[key]) for key in keys_to_use] for res in goea_results_all]
	table = pd.DataFrame(lines, columns=keys_to_use)

	#Convert e (enriched)/p (purified) to increased/decreased
	translation_dict = {"e": "increased", "p": "decreased"}
	table.replace({"enrichment": translation_dict}, inplace=True)

	return(table)

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

	1. TF1-TF2:  |---TF1(+)--->   |---TF2(+)--->   =   <TF2-|   <TF1-| 
	2. TF2-TF1:  |TF2+>    |TF1+>   =   <TF1-|   <TF2-| 
	3. convergent: |TF1+>    <TF2-|   =   |TF2+>   <TF1-|
	4. divergent:    <TF1-|    |TF2+>   =   <TF2-|   |TF1+>

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

	"""
	
	#TODO: Test that input is stranded and directional
	
	rules = rules.copy() #ensures that rules-table is not changed
	
	#Split TF names from strands
	rules[["TF1_name", "TF1_strand"]] = rules["TF1"].str.split("(", expand=True)
	rules["TF1_strand"] = rules["TF1_strand"].str.replace(")", "", regex=False)

	rules[["TF2_name", "TF2_strand"]] = rules["TF2"].str.split("(", expand=True)
	rules["TF2_strand"] = rules["TF2_strand"].str.replace(")", "", regex=False)
	
	#Setup count dictionary
	keys = tuples = [tuple(x) for x in rules[["TF1_name", "TF1_strand", "TF2_name", "TF2_strand"]].values]
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

	return(frame)

#-------------------------------------------------------------------------------#
#------------------------------- Network analysis ------------------------------#
#-------------------------------------------------------------------------------#

def _is_symmetric(a, rtol=1e-05, atol=1e-08):
	""" Utility to check if a matrix is symmetric. 
	Source: https://stackoverflow.com/a/42913743
	"""
	return np.allclose(a, a.T, rtol=rtol, atol=atol)

def build_network(table, node1="TF1", node2="TF2", directed=False, multi=False):
	""" 
	Build a networkx network from a table containing node1, node2 and other node/edge attribute columns, e.g. as from CombObj.market_basket() analysis.
	
	Parameters
	----------
	table : pd.DataFrame 
		Edges table including node/edge attributes.
	node1 : str, optional
		The column to use as node1 ID. Default: "TF1".
	node2 : str, optional
		The column to use as node2 ID. Default: "TF2".
	directed : bool, optional
		Whether edges are directed or not. Default: False.
	multi : bool, optional
		Allow multiple edges between two vertices. If false, the first occurrence of TF1-TF2/TF2-TF1 in the table is used. Default: False.

	Returns
	---------
	networkx.Graph / networkx.diGraph / networkx.MultiGraph / networkx.MultiDiGraph - depending on parameters given
	"""
	
	table = table.copy()
	check_columns(table, [node1, node2])

	# Subset edges based on multi
	if multi == False:

		table.set_index([node1, node2], inplace=True)
		pairs = table.index

		#Collect unique pairs (first occurrence is kept)
		to_keep = {}
		for pair in pairs:
			if not pair[::-1] in to_keep: #if opposite was not already found
				to_keep[pair] = ""

		#Subset table
		table = table.loc[list(to_keep.keys())]
		table.reset_index(inplace=True)

	######### Setup node attributes #########
	attribute_columns = [col for col in table.columns if col not in [node1, node2]]
	sub = table[:100000] #subset in interest of performance
	factorized = sub.apply(lambda x : pd.factorize(x)[0]) + 1 #factorize to enable correlation

	#Establish node1 and node2 attributes
	node_attributes = {}
	columns_to_assign = attribute_columns[:]
	for node in [node1, node2]:
		node_attributes[node] = []
		
		for attribute in columns_to_assign:
			p = scipy.stats.chisquare(factorized[node], f_exp=factorized[attribute])[1]
			if p == 1.0: #columns are fully correlated; save to node attribute
				node_attributes[node] += [attribute]
		
		#Remove attributes from columns_to_assign (prevents the same columns from being assigned to both TF1 and TF2)
		for att in node_attributes[node]:
			columns_to_assign.remove(att)
	
	#Setup tables for node1 and node2 information
	node1_attributes = node_attributes[node1]
	node2_attributes = node_attributes[node2]

	node1_table = table[[node1] + node1_attributes].set_index(node1)
	node2_table = table[[node2] + node2_attributes].set_index(node2)

	#Merge node information to dict for network
	node_table = node1_table.merge(node2_table, left_index=True, right_index=True, how="outer")
	node_table.fillna(0, inplace=True)
	node_attributes = node_table.columns
	node_attribute_dict = {i: {att: row[att] for att in node_attributes} for i, row in node_table.iterrows()}

	######## Setup edge attributes #######	
	edge_attributes = [col for col in attribute_columns if col not in node_attribute_dict]
	edges = [(row[node1], row[node2], {att: row[att] for att in edge_attributes}) for i, row in table.iterrows()]
		
	############ Setup Graph ############
	
	if multi == True:
		if directed == True:
			G = nx.MultiDiGraph()
		else:
			G = nx.MultiGraph()
	else:
		if directed == True:
			G = nx.diGraph()
		else:
			G = nx.Graph()

	#Add collected edges
	G.add_edges_from(edges)
	
	#Add node attributes
	nx.set_node_attributes(G, node_attribute_dict)

	return(G)


def get_degree(G, weight=None):
	"""
	Get degree per node in graph. If weight is given, the degree is the sum of weighted edges.

	Parameters
	-----------
	G : networkx.Graph

	weight : str, optional
		Name of an edge attribute within network. Default: None

	Returns
	--------
	DataFrame

	"""
	
	if weight is None:
		unweighted = dict(G.degree())
		df = pd.DataFrame.from_dict(unweighted, orient="index")
		
	else:
		edge_attributes = list(list(G.edges(data=True))[0][-1].keys())
		if weight in edge_attributes:
			weighted = dict(G.degree(weight=weight))
			df = pd.DataFrame.from_dict(weighted, orient="index")
		else:
			raise ValueError("Weight '{0}' is not an edge attribute of given network. Available attributes are: {1}".format(weight, edge_attributes))
	
	df.columns = ["degree"] 
	df.sort_values("degree", inplace=True, ascending=False)    

	return(df)

#.group_TFs (based on network)
	#although it is only pairs, hubs can be found of TF-cocktails. 

#Center-piece subgraphs

#Graph partitioning 
def get_partitions(network):
	"""
	Partition a network using community louvain.

	Parameters
	----------
	network : networkx.Graph 

	"""

	#network must be undirected
	#if graph.is_directed():
	#raise TypeError("Bad graph type, use only non directed graph")


	partition_dict = community_louvain.best_partition(network)
	partition_dict_fmt = {key:{"partition": value} for key, value in partition_dict.items()}

	#Add partition information to each node
	nx.set_node_attributes(network, partition_dict_fmt)

	return(partition)



#-------------------------------------------------------------------------------#
#----------------------------- Annotation of sites -----------------------------#
#-------------------------------------------------------------------------------#

def annotate_peaks(regions, gtf, config=None):
	"""
	Annotate regions with genes from .gtf using UROPA _[1]. 

	Parameters
	----------
	regions : tobias.utils.regions.RegionList()
		A RegionList object
	gtf : str
		path to gtf file
	config : dict


	Returns
	--------
	None



	Reference
	----------
	.. [1] Kondili M, Fust A, Preussner J, Kuenne C, Braun T, and Looso M. 
	UROPA: a tool for Universal RObust Peak Annotation. Scientific Reports 7 (2017), doi: 10.1038/s41598-017-02464-y
	
	"""
	
	#setup logger
	logger = logging.getLogger("logger")
	
	cfg_dict = {"queries": [{"distance": [10000, 1000], "feature_anchor": "start", "feature": "gene"}],
				"priority": True, 
				"show_attributes": "all"}
	
	cfg_dict = format_config(cfg_dict, logger=logger)
	#print(cfg_dict)
	
	#Convert peaks to dict for uropa
	region_dicts = []
	for region in regions:
		d = {"peak_chr": region.chrom, 
			 "peak_start": region.start, 
			 "peak_end": region.end, 
			 "peak_id": region.name,
			 "peak_score": region.score,
			 "peak_strand": region.strand}
		region_dicts.append(d)
	
	#Index tabix
	if not os.path.exists(gtf + ".tbi"):
		pysam.tabix_index(gtf, preset="gff", keep_original=True)
	
	#Open tabix file
	tabix_obj = pysam.TabixFile(gtf) #, index=gtf_index)

	#For each peak in input peaks, collect all_valid_annotations
	#logger.debug("Annotating peaks in chunk {0}".format(idx))
	all_valid_annotations = []
	for region in region_dicts:
		
		#Annotate single peak
		valid_annotations = annotate_single_peak(region, tabix_obj, cfg_dict, logger=logger)
		all_valid_annotations.extend(valid_annotations)

	tabix_obj.close()
	
	best_annotations = [region for region in all_valid_annotations if region.get("best_hit", 0) == 1]
	
	#Add information to .annotation for each peak
	for i, region in enumerate(regions):
		region.annotation = best_annotations[i]
	
	#Return None; alters peaks in place
