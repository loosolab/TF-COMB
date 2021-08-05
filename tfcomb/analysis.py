from __future__ import print_function

import os
import pkg_resources
import pandas as pd
import numpy as np
import itertools
import scipy
from tfcomb.utils import check_columns
from scipy.stats import chisquare
import re
import graph_tool.all
import copy
import multiprocessing as mp

#Network analysis
import networkx as nx
import community as community_louvain

#GO-term analysis
from goatools.base import download_ncbi_associations
from goatools.base import download_go_basic_obo
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
from goatools.obo_parser import GODag

#Internal functions
from tfcomb.logging import TFcombLogger
from tfcomb.utils import check_columns

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


def _establish_node_attributes(table, node):
	""" 
	Returns a list of columns fitting to 'node' values.

	Parameters
	----------
	table : pd.DataFrame 
		Edges table including node/edge attributes.
	node : str
		Name of column contain node names.

	Returns
	-------
	list
		List of column names
	"""	

	node_attributes = []
	columns_to_assign = [col for col in table.columns if col != node]

	sub = table[:100000] #subset in interest of performance
	factorized = sub.apply(lambda x : pd.factorize(x)[0]) + 1 #factorize to enable correlation

	for attribute in columns_to_assign:
		p = scipy.stats.chisquare(factorized[node], f_exp=factorized[attribute])[1]
		if p == 1.0: #columns are fully correlated; save to node attribute
			node_attributes.append(attribute)
	
	return(node_attributes)


def build_gt_network(table, 
					node1="TF1", 
					node2="TF2", 
					node_table=None,
					directed=False, verbosity=1, 
					node1_attributes=None, 
					node2_attributes=None):
	""" Build graph-tool network from table 
	
	Parameters
	-----------
	table : pd.DataFrame 
		Edges table including node/edge attributes.
	node1 : str, optional
		The column to use as node1 ID. Default: "TF1".
	node2 : str, optional
		The column to use as node2 ID. Default: "TF2".
	directed : bool, optional
		Whether edges are directed or not. Default: False.
	verbosity : int, optional
		Verbosity of logging (0/1/2/3). Default: 1.
	node1_attributes : list, optional
		A list of columns to use for node1 attributes. Default: is estimated from data.
	node2_attributes : list, optional
		A list of columns to use for node2 attributes. Default: is estimated from data.

	"""

	#Setup logger
	logger = TFcombLogger(verbosity)

	#Setup graph
	g = graph_tool.all.Graph(directed=directed)

	#Get dtypes of all attributes
	columns = table.columns
	column_dtypes = table.dtypes.values
	dtype_list = [re.sub(r'[0-9]+', '', str(dtype)) for dtype in column_dtypes]
	dtype_list = [dtype if dtype != "object" else "string" for dtype in dtype_list]
	dtype_dict = dict(zip(columns, dtype_list))
	logger.debug("'dtype_dict': {0}".format(dtype_dict))

	## Setup node table
	if node_table is None:

		node1_attributes = _establish_node_attributes(table, node1)
		node2_attributes = _establish_node_attributes(table, node2)
		node1_table = table[[node1] + node1_attributes].set_index(node1, drop=False)
		node2_table = table[[node2] + node2_attributes].set_index(node2, drop=False)
		node_table = node1_table.merge(node2_table, left_index=True, right_index=True, how="outer")
		node_table.fillna(0, inplace=True)
		node_table.drop_duplicates(inplace=True)
		logger.spam("'node_table': {0}".format(node_table.head()))

	### Setup node/edge attributes ###
	## node attributes
	
	
	node_attributes = list(set(node1_attributes + node2_attributes))
	for att in node_attributes:	#check that node attributes are in table
		if att not in columns:
			raise ValueError("Given node attribute '{0}' is not in table columns.")

	node_attributes = [node1, node2] + node_attributes
	logger.debug("Node attributes: {0}".format(node_attributes))
	for att in node_attributes:
		eprop = g.new_vertex_property(dtype_dict[att])
		g.vertex_properties[att] = eprop

	## edge attributes - the remaining columns
	edge_attributes = list(set(columns) - set(node_attributes))
	for att in edge_attributes:
		if att not in columns:
			raise ValueError("Given edge attribute '{0}' is not in table columns.") 

	logger.debug("Edge attributes: {0}".format(edge_attributes))
	for att in edge_attributes:
		eprop = g.new_edge_property(dtype_dict[att])
		g.edge_properties[att] = eprop

	### Build network ###

	

	## Add nodes with properties
	name2idx = {} #TF name to idx
	idx2name = {}
	for i, row in node_table.to_dict(orient="index").items():
		v = g.add_vertex()
		
		name = i #index is the name of node (TF)
		name2idx[name] = v #idx of node
		idx2name[int(v)] = name
		
		for prop in g.vertex_properties:
			g.vertex_properties[prop][v] = row[prop]

	## Add edges with properties
	for i, row in table.to_dict(orient="index").items(): #loop over all edges in table
		v1, v2 = name2idx[row[node1]], name2idx[row[node2]]
		e = g.add_edge(v1, v2)
		
		for prop in g.edge_properties:
			g.edge_properties[prop][e] = row[prop]
		
	return(g)

def build_nx_network(edge_table, 
						node1="TF1", 
						node2="TF2", 
						node_table=None,
						directed=False, 
						multi=False, 
						verbosity=1 
						):
	""" 
	Build a networkx network from a table containing node1, node2 and other node/edge attribute columns, e.g. as from CombObj.market_basket() analysis.
	
	Parameters
	----------
	edge_table : pd.DataFrame 
		Edge table including node1/node2 attributes.
	node1 : str, optional
		The column within edges_table to use as node1 ID. Default: "TF1".
	node2 : str, optional
		The column within edges_table to use as node2 ID. Default: "TF2".
	node_table : pd.DataFrame, optional
		An additional table of attributes to use for nodes. Default: node attributes are estimated from the columns in edge_table.
	directed : bool, optional
		Whether edges are directed or not. Default: False.
	multi : bool, optional
		Allow multiple edges between two vertices. If False, the first occurrence of TF1-TF2/TF2-TF1 in the table is used. Default: False.
	verbosity : int, optional
		Verbosity of logging (0/1/2/3). Default: 1.

	Returns
	---------
	networkx.Graph / networkx.DiGraph / networkx.MultiGraph / networkx.MultiDiGraph - depending on parameters given
	"""
	
	#Setup logger
	logger = TFcombLogger(verbosity)

	#Setup table
	table = edge_table.copy()
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

		logger.spam("Subset edges (head): {0}".format(table.head()))

	######### Setup node attributes #########
	attribute_columns = [col for col in table.columns if col not in [node1, node2]]

	if node_table is None:
		
		#Establish node attributes
		node1_attributes = _establish_node_attributes(table, node1)
		logger.debug("node1_attributes: {0}".format(node1_attributes))
		
		node2_attributes = _establish_node_attributes(table, node2)
		node2_attributes = list(set(node2_attributes) - set(node1_attributes)) #prevent the same columns from being assigned to both TF1 and TF2)
		logger.debug("node2_attributes: {0}".format(node2_attributes))

		#Setup tables for node1 and node2 information
		node1_table = table[[node1] + node1_attributes].set_index(node1, drop=False) #also includes node1
		node2_table = table[[node2] + node2_attributes].set_index(node2, drop=False) #also includes node2

		#Merge node information to dict for network
		node_table = node1_table.merge(node2_table, left_index=True, right_index=True, how="outer")
		#node_table.fillna(0, inplace=True)
		node_table.drop_duplicates(inplace=True)
	
	else:
		#todo: check that node_table fits with index
		pass

	logger.spam("node_table head:\n{0}".format(node_table.head()))
	node_attributes = list(node_table.columns)
	logger.debug("node_attributes: {0}".format(node_attributes))
	node_attribute_dict = {i: {att: row[att] for att in node_attributes} for i, row in node_table.iterrows()}
	logger.spam("node_attribute_dict: {0} (...)".format({i: node_attribute_dict[i] for i in list(node_attribute_dict.keys())[:5]}))

	######## Setup edge attributes #######	
	edge_attributes = [col for col in attribute_columns if col not in node_attribute_dict]
	logger.debug("edge_attributes: {0}".format(edge_attributes))
	edges_list = [(row[node1], row[node2], {att: row[att] for att in edge_attributes}) for i, row in table.iterrows()]
	logger.spam("edges_list: {0} (...)".format(edges_list[:3]))

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
	G.add_edges_from(edges_list)
	
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
		Name of an edge attribute within network. Default: None.

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
def partition_louvain(network):
	"""
	Partition a network using community louvain.

	Parameters
	----------
	network : networkx.Graph 

	"""

	#network must be undirected
	if network.is_directed():
		raise TypeError("Bad graph type, use only non directed graph")

	#Partition network
	partition_dict = community_louvain.best_partition(network)
	partition_dict_fmt = {key:{"partition": str(value + 1)} for key, value in partition_dict.items()}

	#Add partition information to each node
	nx.set_node_attributes(network, partition_dict_fmt)

	#Setup table with partition information
	table = pd.DataFrame.from_dict(partition_dict_fmt, orient="index")

	return(table)


def partition_gt():
	pass


#-------------------------------------------------------------------------------#
#----------------------------- Annotation of sites -----------------------------#
#-------------------------------------------------------------------------------#

def annotate_peaks(regions, gtf, config=None, threads=1, verbosity=1):
	"""
	Annotate regions with genes from .gtf using UROPA _[1]. 

	Parameters
	----------
	regions : tobias.utils.regions.RegionList()
		A RegionList object
	gtf : str
		path to gtf file
	config : dict

	threads : int
		Number of threads to use for multiprocessing. Default: 1.
	verbosity : int
		Level of verbosity of logger. One of 0,1,2

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
	
	if config is None:
		cfg_dict = {"queries": [{"distance": [10000, 1000], 
					"feature_anchor": "start", 
					"feature": "gene"}],
					"priority": True, 
					"show_attributes": "all"}
	else:
		cfg_dict = config

	cfg_dict = copy.deepcopy(cfg_dict) #make sure that config is not being changed in place
	cfg_dict = format_config(cfg_dict, logger=logger)

	#print(cfg_dict)
	logger.debug("Config dictionary: {0}".format(cfg_dict))
	
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
	gtf_index = gtf + ".tbi"
	if not os.path.exists(gtf_index):
		try:
			gtf = pysam.tabix_index(gtf, preset="gff", keep_original=True)
		except OSError: #gtf is already gzipped
			gtf_gz = gtf + ".gz"
			gtf_index = gtf_gz + ".tbi"
			if not os.path.exists(gtf_index):
				gtf_gz = pysam.tabix_index(gtf_gz, preset="gff", keep_original=True)
			gtf = gtf_gz

	#Split input regions into cores
	n_reg = len(region_dicts)
	per_chunk = int(np.ceil(n_reg/float(threads)))
	region_dict_chunks = [region_dicts[i:i+per_chunk] for i in range(0, n_reg, per_chunk)]

	#Calculate annotations for each chunk
	best_annotations = []
	if threads == 1:
		for region_chunk in region_dict_chunks:
			chunk_annotations = _annotate_peaks_chunk(region_chunk, gtf, cfg_dict)
			best_annotations.extend(chunk_annotations)
	else:
		
		#Start multiprocessing pool
		pool = mp.Pool(threads)

		#Start job for each chunk
		jobs = []
		for region_chunk in region_dict_chunks:
			job = pool.apply_async(_annotate_peaks_chunk, (region_chunk, gtf, cfg_dict, ))
			jobs.append(job)
		pool.close()
	
		#TODO: Print progress
		n_done = [job.ready() for job in jobs]

		#Collect results:
		best_annotations = []
		for job in jobs:
			chunk_annotations = job.get()
			best_annotations.extend(chunk_annotations)

		pool.join()
	"""
	#Open tabix file
	tabix_obj = pysam.TabixFile(gtf, index=gtf_index)

	#For each peak in input peaks, collect all_valid_annotations
	#logger.debug("Annotating peaks in chunk {0}".format(idx))
	all_valid_annotations = []
	n_regions = len(region_dicts)
	n_progress = int(n_regions / 10)
	for i, region in enumerate(region_dicts):
		
		#Print out progress
		if i + 1 % n_progress == 0:
			logger.info("Progress: {0}/{1} regions annotated".format(i+1, n_regions))

		#Annotate single peak
		valid_annotations = annotate_single_peak(region, tabix_obj, cfg_dict, logger=logger)
		all_valid_annotations.extend(valid_annotations)

	tabix_obj.close()
	
	best_annotations = [region for region in all_valid_annotations if region.get("best_hit", 0) == 1]
	"""
	#Add information to .annotation for each peak
	for i, region in enumerate(regions):
		region.annotation = best_annotations[i]
	
	logger.info("Attribute '.annotation' was added to each region")
	#Return None; alters peaks in place

def _annotate_peaks_chunk(region_dicts, gtf, cfg_dict):
	""" Multiprocessing safe function to annotate a chunk of regions """

	logger = logging.getLogger("logger")

	#Open tabix file
	tabix_obj = pysam.TabixFile(gtf)

	#For each peak in input peaks, collect all_valid_annotations
	all_valid_annotations = []
	n_regions = len(region_dicts)
	n_progress = int(n_regions / 10)
	for i, region in enumerate(region_dicts):
	
		#Annotate single peak
		valid_annotations = annotate_single_peak(region, tabix_obj, cfg_dict, logger=logger)
		all_valid_annotations.extend(valid_annotations)

	tabix_obj.close()
	
	best_annotations = [region for region in all_valid_annotations if region.get("best_hit", 0) == 1]

	return(best_annotations)
