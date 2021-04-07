from __future__ import print_function

import os
import pkg_resources
import pandas as pd

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



#----------------------------- Network analysis -----------------------------#










#----------------------------- Annotation of sites -----------------------------#

def annotate_peaks(regions, gtf, config=None, copy=None):
	"""
	
	Parameters
	----------
		regions : RegionList() obj
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


