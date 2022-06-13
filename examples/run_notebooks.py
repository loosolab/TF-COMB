#!/usr/bin/env python
#Script to run all notebooks in the examples folder

import os
import sys
import glob
import subprocess

notebook_dir = os.path.dirname(__file__)

#Get names of all notebooks
notebooks = glob.glob(os.path.join(notebook_dir, "*.ipynb")) 
print("Identified notebooks:")
print(notebooks)

#Set the target order for notebook execution
notebook_order = ["chipseq_analysis.ipynb", "Select_rules.ipynb", "TFBS_from_motifs.ipynb", "Annotate_TFBS.ipynb", "Differential_analysis.ipynb"] 

order_dict = {name: i for i, name in enumerate(notebook_order)}
notebooks = sorted(notebooks, key= lambda x: order_dict.get(os.path.basename(x), 10**10))

#These notebooks must be run manually due to network plotting and errors shown
excluded = ["network_analysis.ipynb", "Differential_analysis.ipynb", "Select_rules.ipynb"]
notebooks = [notebook for notebook in notebooks if notebook not in excluded]

print("Notebooks to run (in order):")
print(notebooks)

#Run notebooks
for notebook in notebooks:
	print("-"*20 + f" executing {notebook} " + "-"*20)
	cmd = "jupyter nbconvert --to notebook --execute --inplace --ClearMetadataPreprocessor.enabled=True {0}".format(notebook)

	exit_code = os.system(cmd)

	if exit_code == 0:
		print("Successfully executed notebook!")
	else:
		print("Errors executing notebook - see above.")
		sys.exit()


