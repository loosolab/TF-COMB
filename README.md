# TF-COMB

TF-COMB stands for “**T**ranscription **F**actor **C**o-**O**ccurrence using **M**arket **B**asket analysis” and is a python module for identifying co-occurring TFs in regulatory regions.

<img align="right" width=200 src="docs/_figures/tfcomb_logo.png">

With a flexible input of ChIP-seq peaks, motif positions, footprint locations, ATAC-seq peaks etc., TF-COMB utilizes a modified market basket analysis to identify TFs (or regions) which are highly co-occurring. The identified TF pairs can then be analyzed in more detail using downstream analysis such as:
- Preferred distance between TFs
- Orientation of stranded regions
- Differential co-occurrence between conditions
- Network analysis to identify protein hubs


Please visit the [Examples](https://tf-comb.readthedocs.io/en/latest/examples/index.html) to see modes of usage.

## Installation
```
$ mamba create -n tfcomb_env --file required_packages.txt
$ conda activate tfcomb_env 
$ pip install .
```  

## Usage
Please visit the full documentation at: [tf-comb.readthedocs.io](https://tf-comb.readthedocs.io)
