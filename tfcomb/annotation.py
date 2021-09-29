import pandas as pd
pd.options.mode.chained_assignment = None #suppress 'SettingWithCopyWarning' prints

#UROPA annotation
import logging
import pysam
from uropa.annotation import annotate_single_peak
from uropa.utils import format_config

#GO-term analysis
from goatools.base import download_ncbi_associations
from goatools.base import download_go_basic_obo
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
from goatools.obo_parser import GODag

#Import tfcomb
import tfcomb.utils
from tfcomb.logging import TFcombLogger

#Load internal data
import pkg_resources
DATA_PATH = pkg_resources.resource_filename("tfcomb", 'data/')

name_to_taxid = {"human": 9606, 
				 "mouse": 10090}



#-------------------------------------------------------------------------------#
#----------------------------- Annotation of sites -----------------------------#
#-------------------------------------------------------------------------------#

def annotate_peaks(regions, gtf, config=None, threads=1, verbosity=1):
	"""
	Annotate regions with genes from .gtf using UROPA _[1]. 

	Parameters
	----------
	regions : tobias.utils.regions.RegionList()
		A RegionList object with positions of genomic elements e.g. TFBS.
	gtf : str
		Path to .gtf file
	config : dict
		A dictionary indicating how regions should be annotated. Default is to annotate feature 'gene' within -10000;1000bp of the gene start. See 'Examples' of how to set up a custom configuration dictionary.
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


	Examples
	---------
	custom_config = {"queries": [{"distance": [10000, 1000], 
								  "feature_anchor": "start", 
					              "feature": "gene"}],
					"priority": True, 
					"show_attributes": "all"}
	"""
	
	#TODO: Check input types
	check_type(regions, [tobias.utils.regions.RegionList()], "regions")


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

#-------------------------------------------------------------------------------#
#------------------------------- GO-term enrichment ----------------------------#
#-------------------------------------------------------------------------------#

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

def go_enrichment(gene_ids, organism="human", background_gene_ids=None, verbosity=1, plot=True):
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
	verbosity : int
		
	plot : bool

	See also
	---------
	tfcomb.plotting.go_bubble

	Returns
	--------
	pd.DataFrame

	Reference
	----------
	https://www.nature.com/articles/s41598-018-28948-z
	
	"""
	
	#verbosity 0/1/2

	#setup logger
	logger = TFcombLogger(verbosity)
	
	#TODO: Gene_ids must be a list
	tfcomb.utils.check_type(gene_ids, [list], "gene_ids")
	
	#Organism must be in human/mouse
	if organism not in name_to_taxid:
		raise ValueError("Organism '{0}' not available. Please choose any of: {1}".format(organism, list(name_to_taxid.keys())))
	else:
		taxid = name_to_taxid[organism]
	
	##### Read data #####

	#Setup GOATOOLS GO DAG
	logger.info("Downloading GO")
	obo_fname = download_go_basic_obo()
	obodag = GODag(obo_fname)
	
	#Setup gene -> GO term associations
	logger.info("Downloading NCBI associations")
	fin_gene2go = download_ncbi_associations()
	logger.debug("fin_gene2go: {0}".format(fin_gene2go))

	logger.info("Gene2GoReader")
	objanno = Gene2GoReader(fin_gene2go, taxids=[taxid])
	logger.debug(objanno)

	ns2assoc = objanno.get_ns2assc()
	#logger.debug(ns2assoc)

	#Read data from package
	gene_table = pd.read_csv(DATA_PATH + organism + "_genes.txt", sep="\t")
	
	##### Setup analysis ####
	#Setup background gene ids 
	if background_gene_ids == None:
		logger.debug("Getting background_gene_ids from gene_table")
		background_gene_ids = list(set(gene_table["GeneID"].tolist())) #unique gene ids from table
	else:

		#Find best-fitting column in gene_table


		pass
		#check if background_gene_ids are in ns2assoc
	
	#Setup goeaobj
	goeaobj = GOEnrichmentStudyNS(
				background_gene_ids, # List of protein-coding genes
				ns2assoc, # geneid/GO associations
				obodag, # Ontologies
				propagate_counts = False,
				alpha = 0.05, # default significance cut-off
				methods = ['fdr_bh'],
				prt=None, 
				log=None) # defult multipletest correction method
	

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
	
	logger.debug("Running .run_study():")
	if verbosity <= 1:
		goea_results_all = goeaobj.run_study(gene_ids_entrez, prt=None)
	else:
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

def match_column():
	pass