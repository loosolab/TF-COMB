from __future__ import print_function

import os
import pkg_resources
import pandas as pd
import numpy as np
import itertools

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
	Perform go_enrichment on the gene ids given

	Parameters
	-----------
	gene_ids : list
		A list of gene ids
	organism : :obj:`str`, optional
		The organism. Defaults to "human".
	background_gene_ids : list
		Defaults to uniprot proteins.

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

#------------------------ Directionality analysis ---------------------------#


def directionality_analysis():
	""" 
	Perform directionality analysis on the TF pairs in edges table 

	1. TF1-TF2: |TF1+>    |TF2+>   =   <TF2-|   <TF1-|
	2. TF2-TF1: |TF2+>    |TF1+>   =   <TF1-|   <TF2-|
	3. against: |TF1+>    <TF2-|   =   |TF2+>   <TF1-|
	4. away:    <TF1-|    |TF2+>   =   <TF2-|   |TF1+>
	
	"""

	#4 different scenarios
	table[["TF1_name", "TF1_strand"]] = table["TF1"].str.split("(", expand=True)
	table["TF1_strand"] = table["TF1_strand"].str.replace(")", "")

	table[["TF2_name", "TF2_strand"]] = table["TF2"].str.split("(", expand=True)
	table["TF2_strand"] = table["TF2_strand"].str.replace(")", "")


	#Convert counts to dictionary:
	keys = table[["TF1_name", "TF1_strand","TF2_name", "TF2_strand"]].apply(tuple, axis=1) 
	values = table[["TF1_TF2_count","TF1_count","TF2_count"]].apply(np.array, axis=1) 
	pair_dict = dict(zip(keys, values))

	TFs = set(table["TF1_name"].tolist() + table["TF2_name"].tolist())

	for TF1 in TFs:
		for TF2 in TFs:
			for TF1_strand in ["+","-"]:
				for TF2_strand in ["+","-"]:
					key = (TF1, TF1_strand, TF2, TF2_strand)
					if not key in pair_dict:
						pair_dict[key] = np.array([0,0,0])

	null = np.array([0,0,0])
	lines = []
	for pair in itertools.combinations(TFs, 2):
		
		TF1, TF2 = pair
		
		#Scenario 1
		keys = [(TF1, "+", TF2, "+"), (TF2, "-", TF1, "-")] 
		arr = np.sum([pair_dict.get(key, null) for key in keys], axis=0) #list of values
		sce1 = pd.Series(arr, index=["TF1_TF2_count", "TF1_count", "TF2_count"])
		sce1 = market_basket(sce1)
		
		#Scenario 2
		keys = [(TF2, "+", TF1, "+"), (TF1, "-", TF2, "-")]
		arr = np.sum([pair_dict.get(key, null) for key in keys], axis=0) #list of values
		sce2 = pd.Series(arr, index=["TF1_TF2_count", "TF1_count", "TF2_count"])
		sce2 = market_basket(sce2)
		
		#Scenario 3
		keys = [(TF1, "+", TF2, "-"), (TF2, "+", TF1, "-")]
		arr = np.sum([pair_dict.get(key, null) for key in keys], axis=0) #list of values
		sce3 = pd.Series(arr, index=["TF1_TF2_count", "TF1_count", "TF2_count"])
		sce3 = market_basket(sce3)
		
		#Scenario 4
		keys = [(TF1, "-", TF2, "+"), (TF2, "-", TF1, "+")]
		arr = np.sum([pair_dict.get(key, null) for key in keys], axis=0) #list of values
		sce4 = pd.Series(arr, index=["TF1_TF2_count", "TF1_count", "TF2_count"])
		sce4 = market_basket(sce4)
		
		## Calculate variance between lifts
		lifts = [series["lift"] for series in [sce1, sce2, sce3, sce4]]
		total = sum([series["TF1_TF2_count"] for series in [sce1, sce2, sce3, sce4]])
		
		line = [TF1, TF2] + lifts + [total]
		lines.append(line)

	frame = pd.DataFrame(lines)
	frame.fillna(0, inplace=True)



#----------------------------- Network analysis -----------------------------#

#tfcomb.analysis.build_network(edges_table) #, node1=node1, node2=node2)
#edges_to_network()

def _is_symmetric(a, rtol=1e-05, atol=1e-08):
	""" Utility to check if a matrix is symmetric. 
	Source: https://stackoverflow.com/a/42913743
	"""
	return np.allclose(a, a.T, rtol=rtol, atol=atol)



def build_network(table, weight="cosine", multi=True):
	""" 
	Table is the .table from .market_basket()
	
	multi : bool
		Allow multiple edges between two vertices. If false, 
	"""
	
	table = table.copy()
	
	#Find out if table is undirected or directed in terms of weight
	pivot = pd.pivot_table(table, values=weight, index="TF1", columns="TF2")
	matrix = np.nan_to_num(pivot.to_numpy())
	symmetric = _is_symmetric(matrix) #if table is symmetric, the network is undirected
	
	######### Setup node attributes #########
	attribute_columns = [col for col in table.columns if col not in ["TF1", "TF2"]]
	
	node1_attributes = ["TF1_count", "TF1_support"]
	node2_attributes = ["TF2_count", "TF2_support"]
	node_attributes = node1_attributes + node2_attributes
	
	TF1_table = table[["TF1"] + node1_attributes].drop_duplicates()
	TF1_table.set_index("TF1", inplace=True)
	TF2_table = table[["TF2"] + node2_attributes].drop_duplicates()
	TF2_table.set_index("TF2", inplace=True)
	
	node_table = pd.concat([TF1_table, TF2_table], axis=1)
	node_attribute_dict = {i: {att: row[att] for att in node_attributes} for i, row in node_table.iterrows()}
	
	######## Setup edge attributes #######
	#Remove duplicated TF1-TF2 / TF2-TF1 pairs if matrix is symmetric
	if len(table) > 1 and symmetric == True: #table is always symmetric if there is only one edge; but this is not an issue
		TFs = list(set(table["TF1"]))
		unique_pairs = list(itertools.combinations(TFs, 2))
		table.set_index(["TF1", "TF2"], inplace=True)
		available_keys = list(table.index)
		unique_pairs = [pair for pair in unique_pairs if pair in available_keys] #only use pairs present in table

		table = table.loc[unique_pairs]
		table.reset_index(inplace=True)
	
	edge_attributes = ["TF1_TF2_count", "TF1_TF2_support", "confidence", "lift", "cosine", "jaccard"]
	edges = [(row["TF1"], row["TF2"], {att: row[att] for att in edge_attributes}) for i, row in table.iterrows()]
	for edge in edges:
		edge[-1]["weight"] = edge[-1][weight]
	
	############ Setup Graph ############
	if symmetric == True:
		G = nx.Graph()
		G.add_edges_from(edges)
	else:       
		G = nx.MultiDiGraph()
		G.add_edges_from(edges)
	
	#Add node attributes
	nx.set_node_attributes(G, node_attribute_dict)

	return(G)

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
	partition_dict_fmt = {key: {"partition": value} for key, value in partition_dict.items()}

	#Add partition information to each node
	nx.set_node_attributes(network, partition_dict_fmt)

	#return(partition)



#----------------------------- Annotation of sites -----------------------------#

def annotate_peaks(regions, gtf, config=None, copy=None):
	"""
	Annotate regions with genes from .gtf using UROPA. 



	Parameters
	----------
	regions : tobias.utils.regions.RegionList()
		A RegionList object
	gtf : str
		path to gtf file
	
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


