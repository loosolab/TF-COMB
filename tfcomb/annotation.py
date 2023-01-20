import os
import pandas as pd
pd.options.mode.chained_assignment = None #suppress 'SettingWithCopyWarning' prints
import copy
import os
import numpy as np
import gzip
import shutil
import requests
import tobias
import glob
import multiprocessing as mp
import psutil

#UROPA annotation
import logging
import pysam
import uropa
from uropa.annotation import annotate_single_peak
from uropa.utils import format_config

#GO-term analysis
from goatools.base import download_ncbi_associations
from goatools.base import download_go_basic_obo
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
from goatools.obo_parser import GODag

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns

#Import tfcomb
import tfcomb
from tfcomb.utils import check_type, check_value, check_string
from tfcomb.logging import TFcombLogger, InputError

#Load internal data
import pkg_resources
DATA_PATH = pkg_resources.resource_filename("tfcomb", 'data/')
gene_tables = glob.glob(DATA_PATH + "*_genes.txt")
available_organisms = [os.path.basename(gene_table.replace("_genes.txt", "")) for gene_table in gene_tables]

taxid_table = pd.read_csv(DATA_PATH + "tax_ids.txt", sep="\t", header=None)
organism_to_taxid = dict(zip(taxid_table[0], taxid_table[1]))

#-------------------------------------------------------------------------------#
#----------------------------- Annotation of sites -----------------------------#
#-------------------------------------------------------------------------------#

def annotate_regions(regions, gtf, config=None, best=True, threads=1, verbosity=1):
	"""
	Annotate regions with genes from .gtf using UROPA [1]_. 

	Parameters
	----------
	regions : tobias.utils.regions.RegionList() or pandas.DataFrame
		A RegionList object with positions of genomic elements e.g. TFBS or a DataFrame containing chr/start/stop-coordinates. If DataFrame, the function assumes that the order of columns is: 
		'chromosome', 'start', 'end', 'id', 'score', 'strand'.
	gtf : str
		Path to .gtf file containing genomic elements for annotation.
	config : dict, optional
		A dictionary indicating how regions should be annotated. Default is to annotate feature 'gene' within -10000;1000bp of the gene start. See 'Examples' of how to set up a custom configuration dictionary.
	best : boolean
		Whether to return the best annotation or all valid annotations. Default: True (only best are kept).
	threads : int, optional
		Number of threads to use for multiprocessing. Default: 1.
	verbosity : int, optional
		Level of verbosity of logger. One of 0,1, 2. Default: 1.

	Returns
	--------
	pd.DataFrame or None
		Dataframe including regions and annotation information (if applicable, otherwise a warning will be displayed and None is returned).
		

	References
	----------
		.. [1] Kondili M, Fust A, Preussner J, Kuenne C, Braun T, and Looso M. UROPA: a tool for Universal RObust Peak Annotation. Scientific Reports 7 (2017), doi: 10.1038/s41598-017-02464-y

	Examples
	---------
	>>> custom_config = {"queries": [{"distance": [10000, 1000], 
	...					 "feature_anchor": "start", 
	...					 "feature": "gene"}],
	...					 "priority": True, 
	...					 "show_attributes": "all"}
	
	#Annotate regions (data/ refers to the data directory of the tfcomb github repository) \n
	
	>>> regions = pd.read_csv("data/GM12878_hg38_chr4_ATAC_peaks.bed")				
	>>> annotate_regions(regions, gtf="data/chr4_genes.gtf",
							  config=custom_config)
	"""
	
	#Check input types
	check_type(regions, [list, tobias.utils.regions.RegionList, pd.DataFrame], "regions")
	check_type(gtf, str, "gtf")
	check_type(config, [type(None), dict], "config")
	check_value(threads, vmin=1, name="threads")

	#Setup logger (also checks verbosity is valid)
	logger = TFcombLogger(verbosity)
	
	#Establish configuration dict
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
	logger.debug("Config dictionary: {0}".format(cfg_dict))

	#Convert regions to dict for uropa
	region_dicts = []
	if type(regions) == pd.DataFrame:	
		for idx, row in regions.iterrows():
			elements = row.iloc[:6].tolist()
			d = {"peak_chr": elements[0],
				 "peak_start": int(elements[1]),
				 "peak_end": int(elements[2]),
				 "peak_id": elements[3],
				 "peak_score": elements[4],
				 "peak_strand": elements[5]}
			region_dicts.append(d)
	else:
		for region in regions:
			d = {"peak_chr": region.chrom, 
				"peak_start": region.start, 
				"peak_end": region.end, 
				"peak_id": region.name,
				"peak_score": region.score,
				"peak_strand": region.strand}
			region_dicts.append(d)
	
	#Index tabix
	gtf_given = gtf 	#gtf before indexing
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

	#Force close of gtf file left open; pysam issue 1038
	proc = psutil.Process()
	for f in proc.open_files():
		if f.path == os.path.abspath(gtf_given):
			os.close(f.fd) 

	#Split input regions into cores
	n_reg = len(region_dicts)
	per_chunk = int(np.ceil(n_reg/float(threads)))
	region_dict_chunks = [region_dicts[i:i+per_chunk] for i in range(0, n_reg, per_chunk)]

	#Calculate annotations for each chunk
	annotations = []
	if threads == 1:
		for region_chunk in region_dict_chunks:
			chunk_annotations = _annotate_peaks_chunk(region_chunk, gtf, cfg_dict)
			annotations.extend(chunk_annotations)
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
		for job in jobs:
			chunk_annotations = job.get()
			annotations.extend(chunk_annotations)

		pool.join()

	#Select best annotations
	if best == True:
		annotations = [annotations[i] for i, anno in enumerate(annotations) if anno["best_hit"] == 1]
	
	
	found_annotation = False
	#Extend feat_attributes per annotation and format for output
	del_keys = ["raw_distance", "anchor_pos", "query", "peak_center", "peak_length", "feat_length", "feat_center"] 
	for anno in annotations:
		if "feat_attributes" in anno:
			# at least one annotation was found
			found_annotation = True
			for key in anno["feat_attributes"]:
				anno[key] = anno["feat_attributes"][key]
			del anno["feat_attributes"]

		#remove certain keys
		for key in del_keys:
			if key in anno:
				del anno[key]

		#Remove best_hit column if best is True
		if best == True:
			del anno["best_hit"]

		#Convert any lists to string
		for key in anno:
			if isinstance(anno[key], list):
				anno[key] = anno[key][0]


	# handle no annotations found
	if not found_annotation:
		logger.warning("No annotations found for given config. Return type is None.")
		return None

	#Convert to pandas table
	annotations_table = pd.DataFrame(annotations)

	#Merge original sites with annotations
	if type(regions) == pd.DataFrame:
		region_columns = regions.columns.tolist()[:6]
		annotation_columns = ["peak_chr", "peak_start", "peak_end",	"peak_id", "peak_score", "peak_strand"]
		anno_unique = annotations_table.drop_duplicates()
		annotations_table = regions.merge(anno_unique, left_on=region_columns, right_on=annotation_columns, how="left")
		annotations_table.drop(columns=annotation_columns, inplace=True)

	#Add information to .annotation for each peak
	#for i, region in enumerate(regions):
	#	region.annotation = best_annotations[i]
	#logger.info("Attribute '.annotation' was added to each region")
	
	return(annotations_table)

def _annotate_peaks_chunk(region_dicts, gtf, cfg_dict):
	""" Multiprocessing safe function to annotate a chunk of regions """

	logger = uropa.utils.UROPALogger()

	#Open tabix file
	tabix_obj = pysam.TabixFile(gtf)

	#For each peak in input peaks, collect all_valid_annotations
	all_valid_annotations = []
	for i, region in enumerate(region_dicts):
	
		#Annotate single peak
		valid_annotations = annotate_single_peak(region, tabix_obj, cfg_dict, logger=logger)
		all_valid_annotations.extend(valid_annotations)

	tabix_obj.close()

	return(all_valid_annotations)

def get_annotated_genes(regions, attribute="gene_name"):
	""" Get list of genes from the list of annotated regions from annotate_regions(). 
	
	Parameters
	-----------
	regions : RegionList() or list of OneTFBS objects 
	
	attribute : str
		The name of the attribute in the 9th column of the .gtf file. Default: 'gene_name'.
	"""
	
	genes = []
	for region in regions:
		
		#Check if region has any annotation
		if "feature" in region.annotation: 
			att_dict = region.annotation.get("feat_attributes", {})
			gene = att_dict.get(attribute, [None])[0]
			
			#if gene == None:
			#    print(region.annotation)
			genes.append(gene)
	
	#Format gene list
	genes = [gene for gene in genes if gene is not None] #Remove None from list
	genes = list(set(genes)) #remove duplicates
	
	return(genes)

#-------------------------------------------------------------------------------#
#------------------------------- GO-term enrichment ----------------------------#
#-------------------------------------------------------------------------------#

class GOAnalysis(pd.DataFrame):

	aspect_translation = {"BP": "Biological Process",
						  "MF": "Molecular Function",
						  "CC": "Cellular Component"}
	@property
	def _constructor(self):
		return GOAnalysis

	@staticmethod
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

	@staticmethod
	def _match_column(table, lst):
		""" Returns the column name with the best match to the list of ids/values """

		#Which column from gene_table has the best match?
		match = []
		for column in table.columns:
			tup = (column, sum([value in table[column].tolist() for value in lst]))
			match.append(tup)
		
		id_col = sorted(match, key=lambda t: -t[1])[0][0] #column with best matching IDs

		return(id_col)	

	def enrichment(self, genes, 
						organism="hsapiens", 
						background=None, 
						propagate_counts=True,
						min_depth=1, 
						verbosity=1):
		"""
		Perform a GO-term enrichment based on a list of genes. This is a TF-COMB wrapper for goatools.

		Parameters
		-----------
		gene_ids : list
			A list of gene ids.
		organism : :obj:`str`, optional
			The organism of which the gene_ids originate. Defaults to 'hsapiens'.
		background : list, optional
			A specific list of background gene ids to use. Default: The list of protein coding genes of the 'organism' given. 
		propagate_counts : bool
			Whether to propagate counts up the tree to parent GO's. Default: True.
		min_depth : int
			Minimum depth of GO-terms to show in output table. Default: 1.
		verbosity : int, optional
			Default: 1.

		See also
		---------
		tfcomb.annotation.GOAnalysis.plot_bubble()

		Returns
		--------
		GOAnalysis object containing enrichment results

		Reference
		----------
		https://www.nature.com/articles/s41598-018-28948-z
		
		"""
		
		#verbosity 0/1/2

		#setup logger
		logger = TFcombLogger(verbosity)
		
		#TODO: Gene_ids must be a list
		tfcomb.utils.check_type(genes, [list, set], "gene_ids")
		
		#Organism must be in human/mouse
		if organism not in available_organisms:
			raise InputError("Organism '{0}' not available. Please choose any of: {1}".format(organism, available_organisms))
		taxid = organism_to_taxid[organism]
		logger.info("Running GO-term enrichment for organism '{0}' (taxid: {1})".format(organism, taxid))
		
		############### Read data ###############

		## Setup GOATOOLS GO DAG
		obo_fname = "go-basic.obo"
		if not os.path.isfile(obo_fname):
			logger.info("Downloading ontologies")
			obo_fname = download_go_basic_obo()
		obodag = GODag(obo_fname)
		
		## Setup gene -> GO term associations
		#check if gene2go contains any data; delete if not
		fin_gene2go = "gene2go"
		if os.path.exists(fin_gene2go):
			s = os.path.getsize(fin_gene2go)
			if s == 0:
				logger.warning("gene2go has size 0; deleting the file")
				os.remove(fin_gene2go)

		else: #file does not exist; try downloading
			logger.info("Downloading NCBI associations")
			try:
				fin_gene2go = download_ncbi_associations()
			except:
				logger.warning("An error occurred downloading NCBI associations using goatools")
				logger.warning("TF-COMB will attempt to download and extract the file manually")
				
				url = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz"
				base = os.path.basename(url) #gene2go.gz
				
				logger.debug("Downloading {0}".format(url))
				with open(base, "wb") as f:
					r = requests.get(url)
					f.write(r.content)

				logger.debug("Decompressing {0}".format(base))
				with gzip.open(base, 'rb') as f_in:
					with open(fin_gene2go, 'wb') as f_out:
						shutil.copyfileobj(f_in, f_out)

		logger.debug("fin_gene2go: {0}".format(fin_gene2go))

		logger.debug("Setting up Gene2GoReader")
		objanno = Gene2GoReader(fin_gene2go, taxids=[taxid])
		logger.debug(objanno)

		ns2assoc = objanno.get_ns2assc()
		#logger.debug(ns2assoc)

		
		############### Setup analysis ###############

		#Read data from package
		gene_table = pd.read_csv(DATA_PATH + organism + "_genes.txt", sep="\t")

		###### Setup gene ids ####### 
		logger.info("Setting up gene ids")

		target_col = "entrezgene_id"

		#Check if genes are in ns2assoc; else, try to convert
		all_gene_ids = set(sum([list(ns2assoc[aspect].keys()) for aspect in ns2assoc.keys()], []))
		n_found = sum([gene_id in all_gene_ids for gene_id in genes])
		
		if n_found == 0:
			
			#Which column from gene_table has the best match?
			id_col = self._match_column(gene_table, genes)
			logger.debug("genes best match column: {0}".format(id_col))
			
			#Find out how many ids can be converted
			not_found = set(genes) - set(gene_table[id_col])
			if len(not_found) > 0:
				logger.warning("{0}/{1} gene ids from 'genes' could not be converted to entrez ids for {2}".format(len(not_found), len(set(genes)), organism))

			#Convert to entrez id
			ids2entrez = dict(zip(gene_table[id_col], gene_table[target_col]))
			gene_ids_entrez = [ids2entrez.get(s, -1) for s in genes]
			gene_ids_entrez = set([gene for gene in gene_ids_entrez if gene > 0]) #only genes for which there was a match

			#Save dict for converting back to original values
			entrez2id = dict(zip(gene_table[target_col], gene_table[id_col]))

		else:
			gene_ids_entrez = genes #gene ids were already entrez
			entrez2id = dict(zip(genes, genes))

		###### Setup background gene ids ######
		if background == None:

			#Use data from package
			logger.debug("Getting background_gene_ids from gene_table")
			background_gene_ids_entrez = list(set(gene_table["entrezgene_id"].tolist())) #unique gene ids from table

		else:

			#Find best-fitting column in gene_table
			id_col = self._match_column(gene_table, background)
			ids2entrez = dict(zip(gene_table[id_col], gene_table[target_col]))
			background_gene_ids_entrez = [ids2entrez.get(s, -1) for s in background]
			background_gene_ids_entrez = set([gene for gene in background_gene_ids_entrez if gene > 0]) #only genes for which there was a match
		
		#Check if genes were in ns2assoc




		#Setup goeaobj
		logger.info("Setting up GO enrichment")
		goeaobj = GOEnrichmentStudyNS(
					background_gene_ids_entrez, # List of protein-coding genes in entrezgene format
					ns2assoc, # geneid/GO associations
					obodag, # Ontologies
					propagate_counts = propagate_counts,
					alpha = 0.05, # default significance cut-off
					methods = ['fdr_bh'], # defult multipletest correction method
					prt=None, 
					log=None) 
		
		############### Run study ###############
		logger.debug("Running .run_study():")
		if verbosity <= 1:
			goea_results_all = goeaobj.run_study(gene_ids_entrez, prt=None)
		else:
			goea_results_all = goeaobj.run_study(gene_ids_entrez)
		
		#Convert study items (entrez) to original ids
		for res in goea_results_all:
			res.study_items = [entrez2id[entrez] for entrez in res.study_items]

		############### Format to dataframe ###############

		keys_to_use = ["GO", "name", "NS", "depth", "enrichment", "ratio_in_study", "ratio_in_pop", 
					"p_uncorrected", "p_fdr_bh", "study_count", "study_items"]
		lines = [[self._fmt_field(res.__dict__[key]) for key in keys_to_use] for res in goea_results_all]
		table = pd.DataFrame(lines, columns=keys_to_use)

		#Set minimum depth (stop "propagate" from collecting counts to upper aspect categories)
		table = table[table["depth"] >= min_depth]

		#Convert e (enriched)/p (purified) to increased/decreased
		translation_dict = {"e": "increased", "p": "decreased"}
		table.replace({"enrichment": translation_dict}, inplace=True)

		#Sort by FDR
		table.sort_values(["p_fdr_bh", "study_count"], ascending=[True, False], inplace=True)

		#Convert to GOAnalysis object
		table = GOAnalysis(table)
		logger.debug("Finished go_enrichment!")

		return(table)


	def plot_bubble(self, aspect="BP", n_terms=20, threshold=0.05, title=None, save=None):
		"""
		Plot a bubble-style plot of GO-enrichment results.

		Parameters
		--------------
		aspect : str
			The aspect for which GO-terms should be shown. Must be one of ["BP", "MF", "CC"]. Default: "BP".
		n_terms : int
			Maximum number of terms to show in graph. Default: 20
		threshold : float between 0-1
			FDR threshold for significant GO-terms. Default: 0.05.	
		title : str
			Custom title for the plot. Default: 'GO-terms for <aspect>'.
		save : str, optional
			Save the plot to the file given in 'save'. Default: None.
		
		Returns
		----------
		ax

		"""
		
		check_string(aspect, ["BP", "CC", "MF"], "aspect") #aspect has to be one of {'BP', 'CC', 'MF'}
		
		#Choose aspect
		table = self #self is a subclass of pd.dataframe
		aspect_table = table[table["NS"] == aspect]
		aspect_table.loc[:,"-log(FDR)"] = -np.log(aspect_table["p_fdr_bh"])
		aspect_table.loc[:,"n_genes"] = aspect_table["study_count"]

		#Sort by pvalue and ngenes
		aspect_table = aspect_table.sort_values(["-log(FDR)", "n_genes"], ascending=False)
		aspect_table.loc[:,"-log(FDR)"] = np.round(aspect_table.loc[:,"-log(FDR)"], 1) #round to prevent long floats on figure
		aspect_table = aspect_table.iloc[:n_terms,:] #first n rows

		#Setup plot
		w, h = 6, (n_terms/20)*4.5 #size 6x4.5 for 20 terms
		fig, ax = plt.subplots(figsize=(w,h))

		#Plot enriched terms 
		ax = sns.scatterplot(x="-log(FDR)", 
									y="name",
									size="n_genes",
									#sizes=(20,500),
									#alpha=0.5,
									hue="-log(FDR)",
									data=aspect_table, 
									ax=ax
									)
		
		#Plot adjustments
		aspect_name = self.aspect_translation[aspect]
		ax.set_ylabel(aspect_name)
		ax.axvline(-np.log(threshold), color="red")
		ax.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
		ax.grid()
		ax.set_axisbelow(True) #puts the grid behind data points

		#Set title for plot depending on input
		if title is None:
			ax.set_title("GO-terms for '{0}'".format(aspect_name)) #, pad=20, size=15)
		else:
			ax.set_title(title)

		if save is not None:
			plt.savefig(save, dpi=600, bbox_inches="tight")

		return(ax)

	
	def compare(self, compare_table):
		"""
		IN PROGRESS: Plot a comparison of two GO-term analysis

		Parameters
		-------------
		compare_table : GOAnalysis object
		"""
		pass
	


