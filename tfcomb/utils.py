import sys
import os
from turtle import end_fill
import pandas as pd
import numpy as np
import math
import copy
from copy import deepcopy
import time
import datetime
import scipy.stats
from scipy.signal import find_peaks
import random
import string
import multiprocessing as mp
import importlib

import pysam
import tfcomb
from tfcomb.logging import TFcombLogger, InputError
from tfcomb.counting import count_co_occurrence
from tobias.utils.regions import OneRegion, RegionList
from tobias.utils.motifs import MotifList
from tobias.utils.signals import fast_rolling_math
import pathlib
import pyBigWig
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation
import seaborn as sns
import tqdm
from IPython import display
import warnings

#----------------- Minimal TFBS class based on the TOBIAS 'OneRegion' class -----------------#

class OneTFBS(list):
	""" Collects location information about one single TFBS """

	def __init__(self, lst=[]):
		
		super(OneTFBS, self).__init__(lst)

		bed_fmt = ["chrom", "start", "end", "name", "score", "strand"]

		#Initialize attributes
		for att in bed_fmt:
			setattr(self, att, None)

		#Set attributes from list
		for i in range(min(len(self), len(bed_fmt))):
			setattr(self, bed_fmt[i], self[i])

		#Overwrite attributes with kwargs
		if hasattr(lst, "__dict__"):
			kwargs = lst.__dict__
			for att, value in kwargs.items():
				setattr(self, att, value)

		self.get_width = OneRegion.get_width
	
	def __str__(self):
		elements = [self.chrom, self.start, self.end, self.name, self.score, self.strand]
		elements = [str(element) for element in elements if element is not None]
		return("\t".join(elements))

	def __repr__(self):
		return(self.__str__())


class TFBSPair():
	""" Collects information about a co-occurring pair of TFBS """

	def __init__(self, TFBS1, TFBS2, anchor="inner", simplify=False):
		"""
		
		Parameters
		-----------
		TFBS1 : OneTFBS
			OneTFBS object for the first TFBS.
		TFBS2 : OneTFBS
			OneTFBS object for the second TFBS.
		anchor : str
			Anchor is used to calculate distances. Must be one of "inner", "outer" or "center". Default: "inner".
		simplify : boolean
			Whether to simplify the orientation to "opposite" (convergent/divergent) and "same" (TF1-TF2/TF2-TF1). Default: False.
		"""

		self.site1 = TFBS1 #OneTFBS object
		self.site2 = TFBS2 #OneTFBS object

		#Calculate distance depending on anchor
		if anchor == "inner":
			distance = self.site2.start - self.site1.end
		elif anchor == "outer":
			distance = self.site1.start - self.site2.end
		elif anchor == "center":
			TF1_anchor = (self.site1.start + self.site1.end) / 2
			TF2_anchor = (self.site2.start + self.site2.end) / 2
			distance = 	int(TF2_anchor - TF1_anchor)
		self.distance = distance

		self._set_orientation(simplify=simplify)

	def _set_orientation(self, simplify=True):
		""" Sets the .orientation attribute of pair dependent on the relative orientation of sites in TFBSPair. 
		
		Parameters
		-------------
		simplify : bool
			Whether to simplify the orientation of sites to "opposite" (convergent/divergent) and "same" (TF1-TF2/TF2-TF1). Default: True.
		"""

		#Calculate orientation scenario
		if self.site1.strand == self.site2.strand:

			if self.site1.strand == "+": #site2 strand is the same
				self.orientation = "TF1-TF2"
			elif self.site1.strand == "-":
				self.orientation = "TF2-TF1"
			elif self.site1.strand == ".":
				self.orientation = "NA" #no orientation applicable
			
		else: #Strands are different

			#Calculate based on whether site1/site2 is stranded
			if self.site1.strand not in ["+","-"]:
				if self.site2.strand == "+":
					self.orientation = "away"
				elif self.site2.strand == "-":
					self.orientation = "towards"
				else:
					self.orientation = "NA"
				
			elif self.site2.strand not in ["+", "-"]:
				if self.site1.strand == "+":
					self.orientation = "towards"
				elif self.site1.strand == "-":
					self.orientation = "away"
				else:
					self.orientation = "NA"
			
			else: #both positions are + or -
			
				if self.site1.strand == "+" and self.site2.strand == "-":
					self.orientation = "convergent"
				elif self.site1.strand == "-" and self.site2.strand == "+":
					self.orientation = "divergent"

		#Simplify orientation if chosen
		if simplify == True:
			translation_dict = {"convergent": "opposite", 
								"divergent": "opposite", 
								"TF1-TF2": "same",
								"TF2-TF1": "same"}
					
			self.orientation = translation_dict.get(self.orientation, self.orientation) #translate if possible

	def __str__(self):
		TFBS1 = ",".join([str(getattr(self.site1, col)) for col in ["chrom", "start", "end", "name", "score", "strand"]])
		TFBS2 = ",".join([str(getattr(self.site2, col)) for col in ["chrom", "start", "end", "name", "score", "strand"]])

		s = f"<TFBSPair | TFBS1: ({TFBS1}) | TFBS2: ({TFBS2}) | distance: {self.distance} | orientation: {self.orientation} >"
		return(s)

	def __repr__(self):
		return(self.__str__())


class TFBSPairList(list):
	""" Class for collecting and analyzing a list of TFBSPair objects """
	# init attributes
	_bigwig_path = None
	_plotting_tables = None
	_last_flank = None
	_last_align = None
	
	def as_table(self):
		""" Table representation of the pairs in the list """
		table = []

		for p in self:
			attributes = getAllAttr(p)

			# add site prefixes/ get attributes for both TFs
			site1 = {f"site1_{key}": value for key, value in getAllAttr(attributes.pop("site1")).items()}
			site2 = {f"site2_{key}": value for key, value in getAllAttr(attributes.pop("site2")).items()}
			attributes = {f"site_{key}": value for key, value in attributes.items()}
			
			attributes.update(site1)
			attributes.update(site2)
			
			table.append(attributes)
		
		table = pd.DataFrame(table)

		# convert types to best possible
		# https://stackoverflow.com/a/65915289
		table = table.apply(pd.to_numeric, errors='ignore').convert_dtypes()

		#Sort columns
		col_order = ["site1_chrom", "site1_start", "site1_end", "site1_name", "site1_score", "site1_strand",
					 "site2_chrom", "site2_start", "site2_end", "site2_name", "site2_score", "site2_strand"]
		order_dict = {col: i for i, col in enumerate(col_order)}
		columns = sorted(table.columns, key= lambda x: order_dict.get(x, 10**10))
		table = table[columns]

		return table
		
	def write_bed(self, outfile, fmt="bed", merge=False):
		""" 
		Write the locations of (TF1, TF2) pairs to a bed-file.
		
		Parameters
		------------
		locations : list
			The output of get_pair_locations().
		outfile : str
			The path which the pair locations should be written to.
		fmt : str, optional
			The format of the output file. Must be one of "bed" or "bedpe". If "bed", the TF1/TF2 sites are written individually (see merge option to merge sites). If "bedpe", the sites are written in BEDPE format. Default: "bed".
		merge : bool, optional
			If fmt="bed", 'merge' controls how the locations are written out. If True, will be written as one region spanning TF1.start-TF2.end. If False, TF1/TF2 sites are written individually. Default: False.
		"""
		
		tfcomb.utils.check_string(fmt, ["bed", "bedpe"], "fmt")
		tfcomb.utils.check_type(merge, bool, "merge")

		#Open output file
		try:
			f = open(outfile, "w")
		except Exception as e:
			raise InputError("Error opening '{0}' for writing. Error message was: {1}".format(outfile, e))
		
		#Write locations to file in format 'fmt'
		if fmt == "bed":
			if merge == True:
				starts = [min([l.site1.start, l.site2.start]) for l in self]
				ends = [max([l.site1.end, l.site2.end]) for l in self]
				s = "\n".join(["\t".join([l.site1.chrom, str(starts[i]), str(ends[i]), l.site1.name + "-" + l.site2.name, str(l.distance), "."]) for i, l in enumerate(self)]) + "\n"
			else:
				s = "\n".join(["\t".join([l.site1.chrom, str(l.site1.start), str(l.site1.end), l.site1.name, ".", l.site1.strand]) + "\n" + 
								"\t".join([l.site2.chrom, str(l.site2.start), str(l.site2.end), l.site2.name, ".", l.site2.strand]) for l in self]) + "\n"
				
		elif fmt == "bedpe":
			s = "\n".join(["\t".join([l.site1.chrom, str(l.site1.start), str(l.site1.end),
										l.site2.chrom, str(l.site2.start), str(l.site2.end), 
										l.site1.name + "-" + l.site2.name, str(l.distance), l.site1.strand, l.site2.strand]) for l in self]) + "\n"
		f.write(s)
		f.close()
	
	# ---------------------- override list inherited functions ----------------------
	# This is done so that plotting_tables are reset when changes in list are detected.

	def append(self, element):
		self._plotting_tables = None
		super().append(element)

	def extend(self, l):
		self._plotting_tables = None
		super().extend(l)

	def __add__(self, x):
		return TFBSPairList(super().__add__(x))

	def insert(self, index, object):
		self._plotting_tables = None
		super().insert(index, object)
	
	def remove(self, value):
		self._plotting_tables = None
		super().remove(value)
	
	def pop(self, index=-1):
		self._plotting_table = None
		return super().pop(index)

	def clear(self) -> None:
		self._plotting_tables = None
		super().clear()

	# slicing functions
	def __getitem__(self, key):
		new = super().__getitem__(key)
		if isinstance(new, list):
			return TFBSPairList(new)
		else:
			return new

	def __setitem__(self, index, value):
		self._plotting_tables = None

		super().__setitem__(index, value)

	# display object function
	def __repr__(self):
		return(f"TFBSPairList({super().__repr__()})")

	# ---------------------- plotting related functions ----------------------

	@property
	def bigwig_path(self):
		"""
		Get path to bigwig file.
		"""
		if not self._bigwig_path:
			raise Exception("No path to signal bigwig found. Please set with 'object.bigwig_path = \"path/to/file.bw\"'.")
		return self._bigwig_path

	@bigwig_path.setter
	def bigwig_path(self, path):
		"""
		Set path to bigwig file. Checks for existence.
		"""
		if not os.path.exists(path):
			raise Exception(f"Could not find file! {path}")
		self._plotting_tables = None
		self._bigwig_path = path

	@property
	def plotting_tables(self):
		"""
		Getter for plotting_tables. Will compute if necessary.
		"""
		if not self._plotting_tables:
			self.comp_plotting_tables()
		return self._plotting_tables

	def comp_plotting_tables(self, flank=100, align="center"):
		"""
		Prepare pair and score tables for plotting.

		Parameters
		------------
		flank : int or tuple, default 100
			Window size of TFBSpair. Adds given amount of bases in both directions counted from alignment anchor (see align) between binding sites. Use a tuple of ints to set left and right flank independently.
		align : str, default 'center'
			Position from which the flanking regions are added. Must be one of 'center', 'left', 'right'.
				'center': Midpoint between binding positions (rounded down if uneven).
				'left': End of first binding position in pair.
				'right': Start of second binding position in pair.
		"""
		if align not in ["center", "left", "right"]:
			raise ValueError(f"Parameter 'align' has to be one of ['center', 'left', 'right']. Got '{align}'.")

		# flank tuple
		if not isinstance(flank, tuple):
			flank = (flank, flank)

		# load bigwig file
		signal_bigwig = pyBigWig.open(self.bigwig_path)

		# get pairs as table & sort by distance
		pairs = self.as_table().sort_values(by='site_distance')

		scores = []
		sorted_pairs = []

		for (index, row) in pairs.iterrows():
			# switch pair to always start with same TF
			if row["site1_name"] < row["site2_name"]:
				site1 = ["site1_chrom", "site1_end", "site1_start", "site1_name", "site1_strand"]
				site2 = ["site2_chrom", "site2_start", "site2_end", "site2_name", "site2_strand"]
				row[site1], row[site2] = row[site2].values, row[site1].values

				# get anchor point (center) to which flanks are added
				if align == "center":
					anchor = row["site1_end"] - row["site_distance"] // 2
				elif align == "left":
					anchor = row["site1_end"]
				elif align == "right":
					anchor = row["site2_start"]

				# compute window from anchor point
				row["window_start"] = anchor + flank[0]
				row["window_end"] = anchor - flank[1]

				# compute relative positions
				row["site1_rel_start"] = row["window_start"] - row["site1_start"]
				row["site1_rel_end"] = row["window_start"] - row["site1_end"]

				row["site2_rel_start"] = row["window_start"] - row["site2_start"]
				row["site2_rel_end"] = row["window_start"] - row["site2_end"]

				# fetch scores
				values = signal_bigwig.values(row["site1_chrom"], row["window_end"], row["window_start"])
				values.reverse()
			else:
				# get anchor point (center) to which flanks are added
				if align == "center":
					anchor = row["site1_end"] + row["site_distance"] // 2
				elif align == "left":
					anchor = row["site1_end"]
				elif align == "right":
					anchor = row["site2_start"]

				# compute window from anchor point
				row["window_start"] = anchor - flank[0]
				row["window_end"] = anchor + flank[1]

				# compute relative positions
				row["site1_rel_start"] = row["site1_start"] - row["window_start"]
				row["site1_rel_end"] = row["site1_end"] - row["window_start"]

				row["site2_rel_start"] = row["site2_start"] - row["window_start"]
				row["site2_rel_end"] = row["site2_end"] - row["window_start"]

				values = signal_bigwig.values(row["site1_chrom"], row["window_start"], row["window_end"])

			sorted_pairs.append(row)
			scores.append(values)

		sorted_pairs = pd.DataFrame(sorted_pairs).reset_index(drop=True)
		scores = pd.DataFrame(scores).fillna(value=0)

		#sort by distance, then relative TFBS start
		sorted_pairs.sort_values(by=["site_distance", "site1_rel_start"], inplace=True)
		scores = scores.loc[sorted_pairs.index]

		# warn if the binding site length is not always the same
		if (len(set(sorted_pairs["site1_rel_end"] - sorted_pairs["site1_rel_start"])) > 1 or
			len(set(sorted_pairs["site2_rel_end"] - sorted_pairs["site2_rel_start"])) > 1):
			warnings.warn("Differences in binding site length detected! This can have undesired effects when plotting. Refer to 'CombObj.TFBS_from_motifs(resolve_overlapping)' to solve.")

		self._last_flank = flank
		self._last_align = align
		self._plotting_tables = (sorted_pairs, scores)

	def pairMap(self,
				logNorm_cbar=None,
				show_binding=True,
				flank_plot="strand",
				figsize=(7, 14),
				output=None,
				flank=None,
				align=None,
				alpha=0.7,
				cmap="seismic",
				show_diagonal=True,
				legend_name_score="Bigwig Score",
				xtick_num=10,
				log=np.log1p,
				dpi=300):
		"""
		Create a heatmap of TF binding pairs sorted for distance.

		Parameters
		-----------
			logNorm_cbar : str, default None
				[None, "centerLogNorm", "SymLogNorm"]
				Choose a type of normalization for the colorbar. \n
				SymLogNorm:
					Use matplotlib.colors.SymLogNorm. This does not center to 0 \n
				centerLogNorm:
					Use custom matplotlib.colors.SymLogNorm from stackoverflow. Note this creates a weird colorbar.
			show_binding : bool, default True
				Shows the TF binding positions as a grey background.
			flank_plot : str, default 'strand'
				["strand", "orientation", None]
				Decide if the plots flanking the heatmap should be colored for strand, strand-orientation or disabled.
			figsize : int tuple, default (7, 14)
				Figure dimensions.
			output : str, default None 
				Save plot to given file.
			flank : int or int tuple, default None
				Bases added to both sides counted from center. Forwarded to comp_plotting_tables().
			align : str, default None
				Alignment of pairs. One of ['left', 'right', 'center']. Forwarded to comp_plotting_tables().
			alpha : float, default 0.7
				Alpha value for diagonal lines, TF binding positions and center line.
			cmap : matplotlib colormap name or object, or list of colors, default 'seismic'
				Color palette used in the main heatmap. Forwarded to seaborn.heatmap(cmap)
			show_diagonal : boolean, default True
				Shows diagonal lines for identifying preference in binding distance.
			legend_name_score : str, default 'Bigwig Score'
				Name of the score legend (upper legend).
			xtick_num : int, default 10
				Number of ticks shown on the x-axis. Disable ticks with None or values < 0.
			log : function, default numpy.log1p
				Function applied to each row of scores. Excludes 0 and will use absolute value for negative numbers adding the sign afterwards.
				Use any of the numpy.log functions. For example numpy.log, numpy.log10 or numpy.log1p. None to disable.
			dpi : float, default 300
				The resolution of the figure in dots-per-inch.
		
		Returns
		----------
			matplotlib.gridspec.GridSpec:
				Object containing the finished pairMap.
		"""

		# check parameter values
		if not logNorm_cbar in [None, "centerLogNorm", "SymLogNorm"]:
			raise ValueError(f"Parameter 'logNorm_cbar' has to be one of [None, \"centerLogNorm\", \"SymLogNorm\"]. But found {logNorm_cbar}.")
		if not flank_plot in ["strand", "orientation", None]:
			raise ValueError(f"Parameter 'flank_plot' hat to be one of [\"strand\", \"orientation\", None]. But found {flank_plot}")

		# fixes FloatingPointError: underflow encountered in multiply
		# https://stackoverflow.com/a/61756043
		# should be set by default but for some reason it isn't for matplotlib 3.5.1/ numpy 1.21.5
		np.seterr(under='ignore')

		fig = plt.figure(figsize=figsize, dpi=dpi)

		# compute plotting tables with custom flank
		if not flank is None and flank != self._last_flank or not align is None and align != self._last_align:
			params = {}
			if flank: params["flank"] = flank
			if align: params["align"] = align

			self.comp_plotting_tables(**params)

		# load data
		pairs, scores = self.plotting_tables

		# log scores
		if log:
			def log_row(row):
				out = np.zeros(len(row))
				
				out[row > 0] = log(row[row > 0])
				out[row < 0] = -log(np.abs(row[row < 0]))
				
				return pd.Series(out)

			scores = scores.apply(axis=1, func=log_row)

		# load custom colorbar normalizaition class
		# https://stackoverflow.com/a/65260996
		if logNorm_cbar == "centerLogNorm":
			class MidpointLogNorm(matplotlib.colors.SymLogNorm):
				"""
				Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
				e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))

				All arguments are the same as SymLogNorm, except for midpoint    
				"""
				def __init__(self, lin_thres, lin_scale, midpoint=None, vmin=None, vmax=None):
					self.midpoint = midpoint
					self.lin_thres = lin_thres
					self.lin_scale = lin_scale
					#fraction of the cmap that the linear component occupies
					self.linear_proportion = (lin_scale / (lin_scale + 1)) * 0.5

					matplotlib.colors.SymLogNorm.__init__(self, lin_thres, lin_scale, vmin, vmax)

				def __get_value__(self, v, log_val, clip=None):
					if v < -self.lin_thres or v > self.lin_thres:
						return log_val
					
					x = [-self.lin_thres, self.midpoint, self.lin_thres]
					y = [0.5 - self.linear_proportion, 0.5, 0.5 + self.linear_proportion]
					interpol = np.interp(v, x, y)
					return interpol

				def __call__(self, value, clip=None):
					log_val = matplotlib.colors.SymLogNorm.__call__(self, value)

					out = [0] * len(value)
					for i, v in enumerate(value):
						out[i] = self.__get_value__(v, log_val[i])

					return np.ma.masked_array(out)

		########## define grid ##########
		# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_nested.html
		# grid: [legend_area, plot_area]
		grid = matplotlib.gridspec.GridSpec(nrows=1,
											ncols=2,
											figure=fig,
											width_ratios=[1, 10],
											wspace=0.3)

		# plot area
		plot_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(ncols=3,
																nrows=1,
																width_ratios=[0.5, 10, 0.5],
																subplot_spec=grid[1],
																wspace=0)

		heatmap = fig.add_subplot(plot_grid[1])
		# add flanks if needed
		if flank_plot is not None:
			strand1 = fig.add_subplot(plot_grid[0])
			strand2 = fig.add_subplot(plot_grid[2])

		# legend area
		legend_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(ncols=1, nrows=2, subplot_spec=grid[0])

		# heatmap legend
		heatmap_led = fig.add_subplot(legend_grid[0])
		heatmap_led.set_title(legend_name_score)

		# strand legend
		if flank_plot is not None:
			strand_led = fig.add_subplot(legend_grid[1])
			strand_led.set_title("TF Binding Strand")

		###### define plots ######
		### main heatmap ###
		
		# get normalization function
		if logNorm_cbar == None:
			norm = None
		elif logNorm_cbar == "centerLogNorm":
			norm = MidpointLogNorm(lin_thres=1, lin_scale=1, midpoint=0)
		elif logNorm_cbar == "SymLogNorm":
			norm = matplotlib.colors.SymLogNorm(linthresh=1, linscale=1)
		
		# get center line position
		if self._last_align == "center":
			pos = pairs["site1_rel_end"][0] + pairs["site_distance"][0] // 2
		elif self._last_align == "left":
			pos = pairs["site1_rel_end"][0]
		elif self._last_align == "right":
			pos = pairs["site2_rel_start"][0]

		plot = sns.heatmap(scores, 
						yticklabels=False,
						xticklabels=False,
						cmap=cmap,
						center=None if logNorm_cbar else 0,
						cbar=True,
						cbar_ax=heatmap_led,
						norm=norm,
						ax=heatmap,
						rasterized=True)

		# set title
		name1, name2 = set(pairs["site1_name"]).pop(), set(pairs["site2_name"]).pop()
		heatmap.set_title(f"{name1} <-> {name2}")

		# show evenly spaced xticks
		if xtick_num and xtick_num > 0:
			xtickpositions = np.linspace(0, len(scores.columns), xtick_num, dtype=int)
			xticklabels = np.linspace(-pos, len(scores.columns) - pos, xtick_num, dtype=int)
			# replace nearest tick to 0 with 0
			tick_index = np.absolute(xticklabels).argmin()
			xticklabels[tick_index], xtickpositions[tick_index] = 0, pos

			plot.set_xticks(xtickpositions)
			plot.set_xticklabels(xticklabels)

		# center line
		plot.vlines(x=pos,
					ymin=0, ymax=len(scores),
					linestyles="dashed",
					color="black",
					alpha=alpha,
					linewidth=1)

		# binding sites
		if show_binding:
			for y, (_, row) in enumerate(pairs.iterrows()):
				# left sites
				plot.add_patch(matplotlib.patches.Rectangle(
					xy=(row["site1_rel_start"], y),
					width=row["site1_rel_end"] - row["site1_rel_start"],
					height=1,
					alpha=alpha,
					color="gray",
					edgecolor=None,
					lw=0,
					rasterized=True
				))

				# right sites
				plot.add_patch(matplotlib.patches.Rectangle(
					xy=(row["site2_rel_start"], y),
					width=row["site2_rel_end"] - row["site2_rel_start"],
					height=1,
					alpha=alpha,
					color="gray",
					edgecolor=None,
					lw=0,
					rasterized=True
				))

		# diagonal binding lines
		if show_diagonal:
			linecolor="black"
			linestyle="solid"

			# left
			# start
			plot.axline(
				xy1=(pairs["site1_rel_start"].max(), 0),
				xy2=(pairs["site1_rel_start"].min(), len(scores)),
				color=linecolor,
				alpha=alpha,
				linestyle=linestyle)
			# end
			plot.axline(
				xy1=(pairs["site1_rel_end"].max(), 0),
				xy2=(pairs["site1_rel_end"].min(), len(scores)),
				color=linecolor,
				alpha=alpha,
				linestyle=linestyle)

			# right
			# start
			plot.axline(
				xy1=(pairs["site2_rel_start"].min(), 0),
				xy2=(pairs["site2_rel_start"].max(), len(scores)),
				color=linecolor,
				alpha=alpha,
				linestyle=linestyle)
			# end
			plot.axline(
				xy1=(pairs["site2_rel_end"].min(), 0),
				xy2=(pairs["site2_rel_end"].max(), len(scores)),
				color=linecolor,
				alpha=alpha,
				linestyle=linestyle)

		# https://moonbooks.org/Articles/How-to-add-a-frame-to-a-seaborn-heatmap-figure-in-python-/
		# make frame visible
		for _, spine in plot.spines.items():
			spine.set_visible(True)

		if flank_plot is not None:
			### strand left ###
			#https://stackoverflow.com/a/57994641
			col = ["site1_strand"] if flank_plot == "strand" else ["site_orientation"]
			
			str_to_int = {j:i for i, j in enumerate(pd.unique(pairs[col[0]].values))}

			cmap = sns.color_palette("tab10", len(str_to_int))

			plot = sns.heatmap(pairs[col].replace(str_to_int),
								yticklabels=False,
								xticklabels=False,
								cmap=cmap,
								cbar=True,
								cbar_ax=strand_led,
								ax=strand1,
								rasterized=True)

			colorbar = plot.collections[0].colorbar 
			r = colorbar.vmax - colorbar.vmin 
			colorbar.set_ticks([colorbar.vmin + r / len(str_to_int) * (0.5 + i) for i in range(len(str_to_int))])
			colorbar.set_ticklabels(list(str_to_int.keys()))

			### strand right ###
			col = ["site2_strand"] if flank_plot == "strand" else ["site_orientation"]
			
			sns.heatmap(pairs[col].replace(str_to_int),
						yticklabels=False,
						xticklabels=False,
						cmap=cmap,
						cbar=False,
						ax=strand2,
						rasterized=True)
		
		# save plot
		if output:
			plt.savefig(output)

		# show plot
		plt.show()

		# close figure
		plt.close()

		return grid

	def pairTrack(self, dist=None, start=None, end=None, ymin=None, ymax=None, ylabel="Bigwig signal", output=None, flank=None, align=None, figsize=(6, 4), dpi=70, _ret_param=False):
		"""
		Create an aggregated footprint track on the paired binding sites.
		Either aggregate all sites for a specific distance or give a range of sites that should be aggregated. 
		If the second approach spans multiple distances the binding locations are shown as a range as well.
		
		
		Parameters
		----------
			dist : int or int list, default None
				Show track for one or more distances between binding sites.
			start : int, default None
				Define start of range of sites that should be aggregated. If set will ignore 'dist'.
			end : int, default None
				Define end of range of sites that should be aggregated. If set will ignore 'dist'.
			ymin : int, default None
				Y-axis minimum limit.
			ymax : int, default None
				Y-axis maximum limit.
			ylabel : str, default 'Bigwig signal'
				Label for the y-axis.
			output : str, default None
				Save plot to given file.
			flank : int or int tuple, default None
				Bases added to both sides counted from center. Forwarded to comp_plotting_tables().
			align : str, default None
				Alignment of pairs. One of ['left', 'right', 'center']. Forwarded to comp_plotting_tables().
			figsize : int tuple, default (3, 3)
				Figure dimensions.
			dpi : float, default 70
				The resolution of the figure in dots-per-inch.
			_ret_param : bool, default False
				Intended for internal animation use!
				If True will cause the function to return a dict of function call parameters used to create plot.
		
		Returns
		----------
			matplotlib.axes._subplots.AxesSubplot or dict:
				Return axes object of the plot.
		"""
		# parameter check
		if dist is None and start is None and end is None:
			raise ValueError("Either set dist or start and end parameter!")
		
		# compute plotting tables with custom flank
		if not flank is None and flank != self._last_flank or not align is None and align != self._last_align:
			params = {}
			if flank: params["flank"] = flank
			if align: params["align"] = align

			self.comp_plotting_tables(**params)

		# load data
		pairs, scores = self.plotting_tables

		# dict holding dict of all function calls to create plot
		parameter = dict()
		
		# get scores
		if not start is None and not end is None:
			if end > len(pairs):
				raise ValueError(f"Out of range! Given {start}:{end} valid range is 0:{len(pairs)}")
			
			tmp_pairs = pairs[start:end]
			tmp_scores = scores[start:end]
			
			dist = f"{tmp_pairs['site_distance'].min()} - {tmp_pairs['site_distance'].max()} | Selected: {start}:{end}"
		else:
			tmp_pairs = pairs[pairs["site_distance"].isin(dist if isinstance(dist, list) else [dist])]
			tmp_scores = scores.loc[tmp_pairs.index] 
		
		# get pair names
		# TODO sanity check for len 1
		lname = set(tmp_pairs["site1_name"]).pop()
		rname = set(tmp_pairs["site2_name"]).pop()

		# compute x axis range (0 centered)
		if self._last_align == "center":
			center = pairs["site1_rel_end"][0] + pairs["site_distance"][0] // 2
		elif self._last_align == "left":
			center = pairs["site1_rel_end"][0]
		elif self._last_align == "right":
			center = pairs["site2_rel_start"][0]

		xmin, xmax = -self._last_flank[0], self._last_flank[1]

		# compute y axis range + 10% padding
		points = tmp_scores.mean()

		if ymin is None:
			ymin = points.min()
			ymin += ymin * 0.1
		if ymax is None:
			ymax = points.max()
			ymax += ymax * 0.1

		
		##### plot #####
		fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
		
		# add aggregated line
		parameter["set"] = {"ylim": (ymin, ymax),
							"xlim": (xmin, xmax),
							"ylabel": ylabel,
							"xlabel": "Basepair",
							"title": f"Distance: {dist} | Sites aggregated: {len(tmp_pairs)}\n{lname} <--> {rname}"}
		
		ax.set(**parameter["set"])
		
		parameter["plot"] = [list(range(xmin, xmax)), points]
		ax.plot(*parameter["plot"])
		
		parameter["vlines"] = {"x": 0,
							"ymin": ymin, 
							"ymax": ymax,
							"linestyles": "dashed",
							"color": "black",
							"alpha": 1,
							"linewidth": 1}
		
		# add center line
		ax.vlines(**parameter["vlines"])
		
		# add binding site locations
		start = tmp_pairs["site1_rel_start"].min() - center
		end = tmp_pairs["site1_rel_end"].max() - center
		parameter["patches.Rectangle"] = [{"xy": (start, ymin),
										"width": end-start,
										"height": np.abs(ymin-ymax),
										"color": 'tab:red',
										"alpha": 0.2}]
		ax.add_patch(matplotlib.patches.Rectangle(**parameter["patches.Rectangle"][0]))
		
		start = tmp_pairs["site2_rel_start"].min() - center
		end = tmp_pairs["site2_rel_end"].max() - center
		parameter["patches.Rectangle"].append({"xy": (start, ymin),
											"width": end-start,
											"height": np.abs(ymin-ymax),
											"color": 'tab:green',
											"alpha": 0.2})
		ax.add_patch(matplotlib.patches.Rectangle(**parameter["patches.Rectangle"][1]))
		
		# save plot
		if output:
			plt.savefig(output)

		# show plot if not in parameter mode
		if not _ret_param:
			plt.show()

		# close figure
		plt.close()
		
		if _ret_param:
			return parameter
		else:
			return ax

	def pairTrackAnimation(self, site_num=None, step=10, ymin=None, ymax=None, ylabel="Bigwig signal", interval=50, repeat_delay=0, repeat=False, output=None, flank=None, align=None, figsize=(6, 4), dpi=70):
		"""
		Combine a set of pairTrack plots to a .gif.
			
		Note
		--------
		The memory limit can be increased with the following if necessary. Default is 20 MB.
		matplotlib.rcParams['animation.embed_limit'] = 100 # in MB
		
		
		Parameters
		----------
			site_num : int, default None
				Number of sites to aggregate for every step. If None will aggregate by distance between binding pair.
			step : int, default None
				Step size between aggregations. Will be ignored if site_num=None.
			ymin : int, default None
				Y-axis minimum limit
			ymax : int, default None
				Y-axis maximum limit
			ylabel : str, default 'Bigwig signal'
				Label for the y-axis.
			interval : int, default 50
				Delay between frames in milliseconds
			repeat_delay : int, default 0
				The delay in milliseconds between consecutive animation runs, if repeat is True.
			repeat : boolean, default False
				Whether the animation repeats when the sequence of frames is completed.
			output : str, default None
				Save plot to given file.
			flank : int or int tuple, default None
				Bases added to both sides counted from center. Forwarded to comp_plotting_tables().
			align : str, default None
				Alignment of pairs. One of ['left', 'right', 'center']. Forwarded to comp_plotting_tables().
			figsize : int tuple, default (6, 4)
				Figure dimensions.
			dpi : float, default 70
				The resolution of the figure in dots-per-inch.
		
		Returns
		----------
			IPython.core.display.HTML:
				Gif object ready to display in a jupyter notebook.
		"""
		# compute plotting tables with custom flank
		if not flank is None and flank != self._last_flank or not align is None and align != self._last_align:
			params = {}
			if flank: params["flank"] = flank
			if align: params["align"] = align

			self.comp_plotting_tables(**params)

		# load data
		pairs, scores = self.plotting_tables

		# prepare plots for animation
		parameter_list = list()
		
		if site_num:
			# compute animation y-range + 10% padding
			if ymin is None:
				ymin = np.min([scores[s:s + site_num if s + site_num < len(pairs) else len(pairs)].mean().min() for s in range(0, len(pairs), step)])
				ymin += ymin * 0.1
			if ymax is None:
				ymax = np.max([scores[s:s + site_num if s + site_num < len(pairs) else len(pairs)].mean().max() for s in range(0, len(pairs), step)])
				ymax += ymax * 0.1

			pbar = tqdm.tqdm(total=len(range(0, len(pairs), step))+1)
			pbar.set_description("Create frames")
			
			for start in range(0, len(pairs), step):
				pbar.update()
				parameter_list.append(self.pairTrack(start=start,
													end=start + site_num if start + site_num < len(pairs) else len(pairs),
													ymin=ymin,
													ymax=ymax,
													ylabel=ylabel,
													_ret_param=True
													)
									)
		else:
			# compute animation y-range + 10% padding
			if ymin is None:
				ymin = np.min([scores.loc[pairs[pairs["site_distance"] == d].index].mean().min() for d in set(pairs["site_distance"])])
				ymin += ymin * 0.1
			if ymax is None:
				ymax = np.max([scores.loc[pairs[pairs["site_distance"] == d].index].mean().max() for d in set(pairs["site_distance"])])
				ymax += ymax * 0.1

			pbar = tqdm.tqdm(total=len(set(pairs["site_distance"]))+1)
			pbar.set_description("Create frames")
			
			for d in set(pairs["site_distance"]):
				pbar.update()
				parameter_list.append(self.pairTrack(dist=d,
													ymin=ymin,
													ymax=ymax,
													ylabel=ylabel,
													_ret_param=True
													)
									)
		# update to be at 100% then close progress bar
		pbar.update()
		pbar.close()
				
		##### Setup animation #####
		# setup figure to draw on
		fig, axes = plt.subplots(figsize=figsize, dpi=dpi)
		line, = axes.plot([])
		axes.add_patch(matplotlib.patches.Rectangle(xy=(0, 0), width=0, height=0))
		axes.add_patch(matplotlib.patches.Rectangle(xy=(0, 0), width=0, height=0))
		
		# progress bar
		pbar = tqdm.tqdm(total=len(parameter_list)+1)
		pbar.set_description("Render animation")
		# create animation function (this decides what is drawn in every frame)
		def animate(i):
			"""
			Draw all elements for a given frame index (i).
			"""
			# update progress bar
			pbar.update()
			
			calls = parameter_list[i]

			for key, params in calls.items():
				if key == "plot":
					line.set_xdata(params[0])
					line.set_ydata(params[1])
				elif key == "patches.Rectangle":
					# remove old patches
					# https://stackoverflow.com/a/62591596
					axes.patches.clear()
					# add new patches
					for p in params:
						axes.add_patch(matplotlib.patches.Rectangle(**p))
				elif isinstance(params, dict):
					eval(f"axes.{key}(**params)")
					
		##### Run animation #####
		anim_created = matplotlib.animation.FuncAnimation(fig, 
														animate,
														frames=len(parameter_list),
														interval=interval,
														repeat_delay=repeat_delay,
														repeat=repeat
														)
		
		# save animation
		if output:
			anim_created.save(output, dpi=dpi)
			pbar.reset()
		
		# prepare output
		video = anim_created.to_jshtml()#to_html5_video()
		html = display.HTML(video)
		
		# close figure & progress bar
		plt.close()
		pbar.close()
		
		return html

	def pairLines(self, x, y, figsize=(6, 4), dpi=70, output=None):
		"""
		Compare miscellaneous values between TF-pair.
		
		Parameters
		----------
			x : string
				Data to show on the x-axis. Set None to get a list of options.
			y : string
				Data to show on the y-axis. Set None to get a list of options.
			figsize : int tuple, default (6, 4)
				Figure dimensions.
			dpi : float, default 70
				The resolution of the figure in dots-per-inch.
			output : str, default None
				Save plot to given file.

		Returns
		----------
			matplotlib.axes._subplots.AxesSubplot:
				Return axes object of the plot.
		"""
		# TODO expose as parameter
		# would need checks for datatype!
		hue="name"

		table = self.as_table()

		# print x, y options
		if not x or not y:
			print(f"x, y options: {set(sl[1] for sl in table.columns.str.split('_', n=1).to_list())}")
			return

		# sort table to always start with the same TF
		if len(set(table["site1_name"])) > 1:
			tf = table["site1_name"][0]
			rf = table["site1_name"] == tf
			
			site1 = table.columns[table.columns.str.startswith("site1")]
			site2 = table.columns[table.columns.str.startswith("site2")]
			
			table.loc[rf, site1.append(site2)] = table.loc[rf, site2.append(site1)].values
		
		# fetch column names
		x_names = list(table.columns[table.columns.str.endswith("_" + x)]) * 2
		y_names = list(table.columns[table.columns.str.endswith("_" + y)]) * 2
		hue_names = list(table.columns[table.columns.str.endswith("_" + hue)]) * 2

		tmp_postfix = True if len(set(hue_names)) >= 2 else False
		
		if not x_names:
			raise Exception(f"Could not find x='{x}'. Available are {set(sl[1] for sl in table.columns.str.split('_', n=1).to_list())}.")
		if not y_names:
			raise Exception(f"Could not find y='{y}'. Available are {set(sl[1] for sl in table.columns.str.split('_', n=1).to_list())}.")
		if not hue_names:
			raise Exception(f"Could not find hue='{hue}'. Available are {set(sl[1] for sl in table.columns.str.split('_', n=1).to_list())}.")

		
		# collect data
		x1, x2 = table[x_names[0]], table[x_names[1]]
		y1, y2 = table[y_names[0]], table[y_names[1]]
		hue1, hue2 = table[hue_names[0]], table[hue_names[1]]
		
		# in case of both names being equal set postfix
		if tmp_postfix:
			hue1, hue2 = hue1 + "_1", hue2 + "_2"
		
		##### plotting #####
		plt.figure(figsize=figsize, dpi=dpi)
		
		plot = sns.lineplot(x=x1.append(x2).values,
							y=y1.append(y2).values,
							hue=hue1.append(hue2).values)
		
		# rotate x-ticks for non numeric data
		if not pd.api.types.is_numeric_dtype(x1.append(x2)):
			plt.xticks(rotation=90)
		
		# remove postfix in legend
		if tmp_postfix:
			handles, labels = plot.get_legend_handles_labels()
			plot.legend(handles=handles, labels=[l[:-2] for l in labels])
		
		# set axis labels
		plot.set(ylabel=y, xlabel=x)
		
		# save plot
		if output:
			plt.savefig(output)

		# show then close figure
		plt.show()
		plt.close()
		
		return plot

	def set_orientation(self, simplify=False):
		""" Fill orientation of each TF pair """

		for pair in self:
			pair._set_orientation(simplify=simplify)


	def plot_distances(self, groupby="orientation", figsize=None, group_order=None):
		""" Plot the distribution of distances between TFBS-pairs.
		
		Parameters
		-----------
		groupby : str
			An attribute of each pair to group distances by. If None, all distances are shown without grouping. Default: "orientation".
		figsize : tuple of ints
			Set the figure size, e.g. (8,10). Default: None (default matplotlib figuresize).
		"""

		#get all possible values in groupby
		#group_values = set([pair.getattr("orientation") for pair in self])

		#Collect distances per group
		distances = {}
		if groupby is not None:
			for pair in self:
				pair_group = getattr(pair, groupby)
				distances[pair_group] = distances.get(pair_group, []) + [pair.distance]
		else:
			distances["all"] = [pair.distance for pair in self]

		#Setup figure
		fig, axarr = plt.subplots(len(distances), sharex=True, sharey=True, figsize=figsize)

		#adjust for one-group plotting
		if len(distances) == 1:
			axarr = [axarr] #make axarr subscriptable

		#Plot distances per group
		if group_order == None:
			group_order = list(distances.keys())
		else:
			#todo: check that groups are in distances
			pass

		for i, group in enumerate(group_order):
			lst = distances[group]
			axarr[i].hist(lst, bins=int(max(lst)) + 1)
			
			axarr[i].set_ylabel("Count")
			axarr[i].text(1, 0.5, f"({group})",
						horizontalalignment='left',
						verticalalignment='center',
						transform=axarr[i].transAxes)

		#Make final adjustments
		for ax in axarr:
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			
		_ = axarr[-1].set_xlabel("Distance")


		return axarr

#------------------------------ Notebook / script exceptions -----------------------------#

def _is_notebook():
	""" Utility to check if function is being run from a notebook or a script """
	try:
		ipython_shell = get_ipython()
		return(True)
	except NameError:
		return(False)

class InputError(Exception):
	""" Raises an InputError exception without writing traceback """

	def _render_traceback_(self):
		etype, msg, tb = sys.exc_info()
		sys.stderr.write("{0}: {1}".format(etype.__name__, msg))

class StopExecution(Exception):
	""" Stop execution of a notebook cell with error message"""

	def _render_traceback_(self):
		etype, msg, _ = sys.exc_info()
		sys.stderr.write("{1}".format(etype.__name__, msg))
		#sys.stderr.write(f"{msg}")

def check_graphtool():
	""" Utility to check if 'graph-tool' is installed on path. Raises an exception (if notebook) or exits (if script) if the module is not installed. """

	error = 0
	try:
		import graph_tool.all
	except ModuleNotFoundError:
		error = 1
	except: 
		raise #unexpected error loading module
	
	#Write out error if module was not found
	if error == 1:
		s = "ERROR: Could not find the 'graph-tool' module on path. This module is needed for some of the TFCOMB network analysis functions. "
		s += "Please visit 'https://graph-tool.skewed.de/' for information about installation."

		if _is_notebook():
			raise StopExecution(s) from None
		else:
			sys.exit(s)
	
	return(True)

def check_module(module):
	""" Check if <module> can be imported without error """

	error = 0
	try:
		importlib.import_module(module)
	except ModuleNotFoundError:
		error = 1
	except: 
		raise #unexpected error loading module
	
	#Write out error if module was not found
	if error == 1:
		s = f"ERROR: Could not find the '{module}' module on path. This module is needed for this functionality. Please install this package to proceed."

		if _is_notebook():
			raise StopExecution(s) from None
		else:
			sys.exit(s)
	
	return(True)

#--------------------------------- File/type checks ---------------------------------#

def check_columns(df, columns):
	""" Utility to check whether columns are found within a pandas dataframe.
	
	Parameters 
	------------
	df : pandas.DataFrame
		A pandas dataframe to check.
	columns : list
		A list of column names to check for within 'df'.

	Raises
	--------
	InputError
		If any of the columns are not in 'df'.
	"""
	
	df_columns = df.columns

	not_found = []
	for column in columns:
		if column is not None:
			if column not in df_columns:
				not_found.append(column)
	
	if len(not_found) > 0:
		error_str = "Columns '{0}' are not found in dataframe. Available columns are: {1}".format(not_found, df_columns)
		raise InputError(error_str)
		
def check_dir(dir_path, create=True):
	""" Check if a dir is writeable.
	
	Parameters
	------------
	dir_path : str
		A path to a directory.
	
	Raises
	--------
	InputError
		If dir_path is not writeable.
	"""
	#Check if dir already exists
	if dir_path is not None: #don't check path given as None; assume that this is taken care of elsewhere
		if os.path.exists(dir_path):
			if not os.path.isdir(dir_path): # is it a file or a dir?
				raise InputError("Path '{0}' is not a directory".format(dir_path))

		#check writeability of parent dir
		else:
			if create:
				pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
			

def check_writeability(file_path):
	""" Check if a file is writeable.
	
	Parameters
	------------
	file_path : str
		A path to a file.
	
	Raises
	--------
	InputError
		If file_path is not writeable.
	"""

	check_type(file_path, str)

	#Check if file already exists
	error_str = None
	if os.path.exists(file_path):
		if not os.path.isfile(file_path): # is it a file or a dir?
			error_str = "Path '{0}' is not a file".format(file_path)

	#check writeability of parent dir
	else:
		pdir = os.path.dirname(file_path)
		pdir = "." if pdir == "" else pdir #if file_path is in current folder, pdir is empty string
		if os.access(pdir, os.W_OK) == False:
			error_str = "Parent directory '{0}' is not writeable".format(pdir)

	#If any errors were found
	if error_str is not None:
		raise InputError(error_str)


def check_type(obj, allowed, name=None):
	"""
	Check whether given object is within a list of allowed types.

	Parameters
	------------
	obj : object
		Object to check type on
	allowed : type or list of types
		A type or a list of object types to be allowed
	name : str, optional
		Name of object to be written in error. Default: None (the input is referred to as 'object')

	Raises
	--------
	InputError
		If object type is not within types.
	"""

	#Convert allowed to list
	if not isinstance(allowed, list):
		allowed = [allowed]

	#Check if any of the types fit
	flag = 0
	for t in allowed:
		if isinstance(obj, t):
			flag = 1

	#Raise error if none of the types fit
	if flag == 0:
		name = "object" if name is None else f'\'{name}\''
		raise InputError("The {0} given has type '{1}', but must be one of: {2}".format(name, type(obj), allowed))

def check_string(astring, allowed, name=None):
	""" 
	Check whether given string is within a list of allowed strings.
	
	Parameters
	------------
	astring : str
		A string to check.
	allowed : str or list of strings
		A string or list of allowed strings to check against 'astring'.
	name : str, optional
		The name of the string to be written in error. Default: None (the value is referred to as 'string').

	Raises
	--------
	InputError
		If 'astring' is not in 'allowed'.
	"""

	#Convert allowed to list
	if not isinstance(allowed, list):
		allowed = [allowed]
	
	#Check if astring is within allowed
	if astring not in allowed:
		name = "string" if name is None else f'\'{name}\''
		raise InputError("The {0} given ({1}) is not valid - it must be one of: {2}".format(name, astring, allowed))

def check_value(value, vmin=-math.inf, vmax=math.inf, integer=False, name=None):
	"""
	Check whether given 'value' is a valid value (or integer) and if it is within the bounds of vmin/vmax.

	Parameters
	-------------
	value : int or float
		The value to check.
	vmin : int or float, optional
		Minimum the value is allowed to be. Default: -infinity (no bound)
	vmax : int or float
		Maxmum the value is allowed to be. Default: +infinity (no bound)
	integer : bool, optional
		Whether value must be an integer. Default: False (value can be float)
	name : str, optional
		The name of the value to be written in error. Default: None (the value is referred to as 'value').

	Raises
	--------
	InputError
		If 'value' is not a valid value as given by parameters.
	"""

	if vmin > vmax:
		raise InputError("vmin must be smaller than vmax")

	error_msg = None
	if integer == True:		
		if not isinstance(value, int):
			error_msg = "The {0} given ({1}) is not an integer, but integer is set to True.".format(name, value)
	else:
		#check if value is any value
		try:
			_ = int(value)
		except:
			error_msg = "The {0} given ({1}) is not a valid number".format(name, value)

	#If value is a number, check if it is within bounds
	if error_msg is None:
		if not ((value >= vmin) & (value <= vmax)):
			error_msg = "The {0} given ({1}) is not within the bounds of [{2};{3}]".format(name, value, vmin, vmax)
	
	#Finally, raise error if necessary:
	if error_msg is not None:
		raise InputError(error_msg)

def random_string(l=8):
	""" Get a random string of length l """
	s = ''.join(random.choice(string.ascii_uppercase) for _ in range(l))
	return(s)

#--------------------------------- Multiprocessing ---------------------------------#

class Progress():

	def __init__(self, n_total, n_print, logger):
		"""
		Utility class to monitor progress of a list of tasks.
		
		Parameters
		-----------
		n_total : int
			Number of total jobs
		n_print : int
			Number of times to write progress
		logger : logger instance
			The logger to use for writing progress  
		"""
		
		self.n_total = n_total
		self.n_print = n_print
		self.logger = logger        

		#At what number of tasks should the updates be written?
		n_step = int(n_total / (n_print))
		self.progress_steps = [n_step*(i+1) for i in range(n_print)]
		
		self.next = self.progress_steps[0] #first limit in progress_steps to write


	def write_progress(self, n_done):
		""" Log the progress of the current tasks.
		
		Parameters
		-----------
		n_done : int
			Number of tasks done (of n_total tasks)
		"""
		
		if n_done >= self.next:
			
			self.logger.info("Progress: {0:.0f}%".format(n_done/self.n_total*100.0))
			
			#in case more than one progress step was jumped
			remaining_steps = [step for step in self.progress_steps if n_done < step] + [np.inf]
			
			self.next = remaining_steps[0]    #this is the next idx to write (or np.inf if end of list was reached)

def log_progress(jobs, logger, n=10):
	""" 
	Log progress of jobs within job list.

	Parameters
	------------
	jobs : list
		List of multiprocessing jobs to write progress for.
	logger : logger instance
		A logger to use for writing out progress.
	n : int, optional
		Maximum number of progress statements to show. Default: 10. 
	"""

	#Setup progress obj
	n_tasks = len(jobs)
	p = Progress(n_tasks, n, logger)

	n_done = sum([task.ready() for task in jobs])
	while n_done != n_tasks:
		p.write_progress(n_done)
		time.sleep(0.1)
		n_done = sum([task.ready() for task in jobs]) #recalculate number of finished jobs

	logger.info("Finished!")

	return(0) 	#doesn't return until the while loop exits


#--------------------------------------- Motif / TFBS scanning and processing ---------------------------------------#

def prepare_motifs(motifs_file, motif_pvalue=0.0001, motif_naming="name"):
	""" Read motifs from motifs_file and set threshold/name. """

	#Read and prepare motifs
	motifs_obj = MotifList().from_file(motifs_file)

	_ = [motif.get_threshold(motif_pvalue) for motif in motifs_obj]
	_ = [motif.set_prefix(motif_naming) for motif in motifs_obj] #using naming from args

	return(motifs_obj)

def open_genome(genome_f):	
	""" Opens an internal genome object for fetching sequences.

	Parameters
	------------
	genome_f : str
		The path to a fasta file.
	
	Returns
	---------
	pysam.FastaFile
	"""

	genome_obj = pysam.FastaFile(genome_f)
	return(genome_obj)

def open_bigwig(bigwig_f):
	"""
	Parameters
	------------
	bigwig_f : str
		The path to a bigwig file.	
	
	"""

	pybw_obj = pyBigWig.open(bigwig_f)

	return(pybw_obj)

def check_boundaries(regions, genome):
	""" Utility to check whether regions are within the boundaries of genome.
	
	Parameters
	-----------
	regions : tobias.utils.regions.RegionList 
		A RegionList() object containing regions to check.
	genome : pysam.FastaFile
		An object (e.g. from open_genome()) to use as reference. 
	
	Raises
	-------
	InputError
		If a region is not available within genome
	"""

	chromosomes = genome.references
	lengths = genome.lengths
	genome_bounds = dict(zip(chromosomes, lengths))

	for region in regions:
		if region.chrom not in chromosomes:
			raise InputError("Region '{0} {1} {2} {3}' is not present in the given genome. Available chromosomes are: {4}.".format(region.chrom, region.start, region.end, region.name, chromosomes))
		else:
			if region.start < 0 or region.end > genome_bounds[region.chrom]:
				raise InputError("Region '{0} {1} {2} {3}' is out of bounds in the given genome. The length of the chromosome is: {4}".format(region.chrom, region.start, region.end, region.name, genome_bounds[region.chrom]))


def unique_region_names(regions):
	""" 
	Get a list of unique region names within regions. 

	Parameters
	-----------
	regions : tobias.utils.regions.RegionList 
		A RegionList() object containing regions with .name attributes.

	Returns
	--------
	list
		The list of sorted names from regions.
	"""

	names_dict = {r.name: True for r in regions}
	names = sorted(list(names_dict.keys()))

	return(names)

def calculate_TFBS(regions, motifs, genome, resolve="merge"):
	"""
	Multiprocessing-safe function to scan for motif occurrences

	Parameters
	----------
	genome : str or 
		If string , genome will be opened 
	regions : RegionList()
		A RegionList() object of regions 
	resolve : str
		How to resolve overlapping sites from the same TF. Must be one of "off", "highest_score" or "merge". If "highest_score", the highest scoring overlapping site is kept.
		If "merge", the sites are merged, keeping the information of the first site. If "off", overlapping TFBS are kept. Default: "merge".

	Returns
	----------
	List of TFBS within regions

	"""

	check_string(resolve, ["merge", "highest_score", "off"], "resolve")

	#open the genome given
	if isinstance(genome, str):
		genome_obj = open_genome(genome)
	else:
		genome_obj = genome

	TFBS_list = RegionList()
	for region in regions:
		seq = genome_obj.fetch(region.chrom, region.start, region.end)
		region_TFBS = motifs.scan_sequence(seq, region)

		#Convert RegionLists to TFBS class
		region_TFBS = RegionList([OneTFBS(region) for region in region_TFBS])

		TFBS_list += region_TFBS

	#Sort all sites
	TFBS_list.loc_sort()

	#Resolve overlapping
	if resolve != "off":
		TFBS_list = resolve_overlaps(TFBS_list, how=resolve)

	if isinstance(genome, str):
		genome_obj.close()

	return(TFBS_list)

def resolve_overlaps(sites, how="merge", per_name=True):
	""" 
	Resolve overlapping sites within a list of genomic regions.

	Parameters
	------------
	sites : RegionList
		A list of TFBS/regions with .chrom, .start, .end and .name information.
	how : str
		How to resolve the overlapping site. Must be one of "highest_score", "merge". If "highest_score", the highest scoring overlapping site is kept.
		If "merge", the sites are merged, keeping the information of the first site. Default: "merge".
	per_name : bool
		Whether to resolve overlapping only per name or across all sites. If 'True' overlaps are only resolved if the name of the sites are equal. 
		If 'False', overlaps are resolved across all sites. Default: True.
	"""
	
	check_string(how, ["highest_score", "merge"], "how")

	#Create a copy of sites to ensure that original sites are not changed
	n_sites = len(sites)
	new_sites = [None]*n_sites
	
	tracking = {} # dictionary for tracking positions of TFBS per name (or across all)
	
	for current_i in range(n_sites):

		current_site = sites[current_i]
		new_sites[current_i] = current_site #might change again during merging

		site_name = current_site.name if per_name == True else "." #control which site to fetch as 'previous'
		
		if site_name in tracking: #if not in tracking, site is the first site of this name
			
			#previous_site = tracking[site_name]["site"]
			previous_i = tracking[site_name]
			previous_site = new_sites[previous_i]

			if (current_site.chrom == previous_site.chrom) and (current_site.start < previous_site.end): #overlapping
								
				#How to deal with overlap:
				if how == "highest_score":
					
					if current_site.score >= previous_site.score: #keep current site
						new_sites[previous_i] = None
						tracking[site_name] = current_i #new tracking
						
					else: #keep previous site
						new_sites[current_i] = None
						#tracking stays the same
						
				elif how == "merge":
					
					merged_end = max([previous_site.end, current_site.end])
					
					#merge site into the previous; keep previous score/strand
					merged = OneTFBS([current_site.chrom, 
									  previous_site.start, 
									  merged_end, 
									  previous_site.name, 
									  previous_site.score,
									  previous_site.strand])
					
					new_sites[previous_i] = merged
					new_sites[current_i] = None
					#tracking i stays the same
					
			else: #no overlaps with previous; save this site to tracking
				tracking[site_name] = current_i
				
		else: #Save first site to tracking
			tracking[site_name] = current_i
	
	resolved = [site for site in new_sites if site is not None]
	
	return(resolved)

def add_region_overlap(a, b, att="overlap"):
	""" Overlap regions in regionlist 'a' with regions from regionlist 'b' and add 
	a boolean attribute to the regions in 'a' containing overlap status with 'b'. 
	
	Parameters
	------------
	a : list of OneTFBS objects
		A list of objects containing genomic locations.
	b : list of OneTFBS objects
		A list of objects containing genomic locations to overlap with 'a' regions.
	att : str, optional
		The name of the attribute to add to 'a' objects. Default: "overlap".
	"""
	
	a = a.copy()
	b = b.copy()
	
	a.sort(key=lambda region: (region.chrom, region.start, region.end))
	b.sort(key=lambda region: (region.chrom, region.start, region.end))
	
	#Establish order of chromosomes
	chromlist = sorted(list(set([region.chrom for region in a] + [region.chrom for region in b])))
	chrom_pos = {chrom: chromlist.index(chrom) for chrom in chromlist}
		
	## Find overlap yes/no
	a_n = len(a)
	b_n = len(b)
	
	a_i = 0 #initialize
	b_i = 0
	while a_i < a_n and b_i < b_n:

		a_chrom, a_start, a_end = a[a_i].chrom, a[a_i].start, a[a_i].end
		b_chrom, b_start, b_end = b[b_i].chrom, b[b_i].start, b[b_i].end
		
		#Check possibility of overlap
		if a_chrom == b_chrom:

			if a_end <= b_start:	#current a is placed before current b
				a_i += 1

			elif a_start >= b_end:	#current a is placed after current b 
				b_i += 1

			else: #a region overlaps b region
				setattr(a[a_i], att, True) #save overlap
				a_i += 1 #see if next a also overlaps this b

		elif chrom_pos[a_chrom] > chrom_pos[b_chrom]: 	#if a_chrom is after current b_chrom
			b_i += 1

		elif chrom_pos[b_chrom] > chrom_pos[a_chrom]:	#if b_chrom is after current a_chrom
			a_i += 1
	
	#The additional sites are False
	for site_a in a:
		if not hasattr(site_a, att):
			setattr(site_a, att, False)
		
	return(a)


#--------------------------------- Background calculation ---------------------------------#

def shuffle_array(arr, seed=1):
	np.random.seed(seed)
	length = arr.shape[0]
	return(arr[np.random.permutation(length),:])

def shuffle_sites(sites, seed=1):
	""" Shuffle TFBS names to existing positions and updates lengths of the new positions.
	
	Parameters
	-----------
	sites : np.array
		An array of sites in shape (n_sites,4), where each row is a site and columns correspond to chromosome, start, end, name.
	
	Returns
	--------
	An array containing shuffled names with site lengths corresponding to original length of sites.
	"""
	
	#Establish lengths of regions
	lengths = sites[:,2] - sites[:,1]
	sites_plus = np.c_[sites, lengths]
	
	#Shuffle names (and corresponding lengths)
	sites_plus[:,-2:] = shuffle_array(sites_plus[:,-2:], seed)
	
	#Adjust coordinates to new length
	#new start = old start + old half length - new half length
	#new end = new start + new length
	sites_plus[:,1] = sites_plus[:,1] + ((sites_plus[:,2] - sites_plus[:,1])/2) - sites_plus[:,-1]/2 #new start
	sites_plus[:,2] = sites_plus[:,1] + sites_plus[:,-1] #new end
	
	#Remove length again
	sites_shuffled = sites_plus[:,:-1]
	
	return(sites_shuffled)

def calculate_background(sites, seed=1, directional=False, **kwargs):
	""" 
	Wrapper to shuffle sites and count co-occurrence of the shuffled sites. 
	
	Parameters
	------------
	sites : np.array
		An array of sites in shape (n_sites,4), where each row is a site and columns correspond to chromosome, start, end, name.
	seed : int, optional
		Seed for shuffling sites. Default: 1.
	directional : bool
		Decide if direction of found pairs should be taken into account. Default: False.
	kwargs : arguments
		Additional arguments for count_co_occurrence
	"""
	
	#Shuffle sites
	s = datetime.datetime.now()
	shuffled = shuffle_sites(sites, seed=seed)
	e = datetime.datetime.now()
	#print("Shuffling: {0}".format(e-s))
	
	s = datetime.datetime.now()
	_, pair_counts = count_co_occurrence(shuffled, **kwargs)
	e = datetime.datetime.now()
	#print("counting: {0}".format(e-s))
	pair_counts = tfcomb.utils.make_symmetric(pair_counts) if directional == False else pair_counts	#Deal with directionality
	
	return(pair_counts)


#--------------------------------- Thresholding ---------------------------------#

def get_threshold(data, which="upper", percent=0.05, _n_max=10000, verbosity=0, plot=False):
	"""
	Function to get upper/lower threshold(s) based on the distribution of data. The threshold is calculated as the probability of "percent" (upper=1-percent).
	
	Parameters
	------------
	data : list or array
		An array of data to find threshold on.
	which : str
		Which threshold to calculate. Can be one of "upper", "lower", "both". Default: "upper".
	percent : float between 0-1
		Controls how strict the threshold should be set in comparison to the distribution. Default: 0.05.
	
	Returns
	---------
	If which is one of "upper"/"lower", get_threshold returns a float. If "both", get_threshold returns a list of two float thresholds.
	"""
	
	distributions = [scipy.stats.norm, scipy.stats.lognorm, scipy.stats.laplace,
					 scipy.stats.expon, scipy.stats.truncnorm, scipy.stats.truncexpon, scipy.stats.wald, scipy.stats.weibull_min]
	
	logger = tfcomb.logging.TFcombLogger(verbosity)

	#Check input parameters
	check_string(which, ["upper", "lower", "both"], "which")
	check_value(percent, vmin=0, vmax=1, name="percent")

	#Subset data to _n_max:
	if len(data) > _n_max:
		np.random.seed(0)
		data = np.random.choice(data, size=_n_max, replace=False)
	
	data_finite = np.array(data)[~np.isinf(data)]

	#Fit data to each distribution
	distribution_dict = {}
	for distribution in distributions:
		
		#Catch any exceptions from fitting
		try:
			params = distribution.fit(data_finite)
		except Exception as e:
			logger.error("Exception ({0}) occurred while fitting data to '{1}' distribution; skipping this distribution. Error message was: {2} ".format(e.__class__.__name__, distribution.name, e))
			continue

		#Test fit using negative loglikelihood function
		mle = distribution.nnlf(params, data_finite)

		#Save info on distribution fit    
		distribution_dict[distribution.name] = {"distribution": distribution,
												"params": params, 
												"mle": mle}
		logger.spam("Fitted data to '{0}' with mle={1} and params: {2}".format(distribution.name, mle, params))
	

	if len(distribution_dict) == 0:
		raise ValueError("No distributions could be fit to the input data.")

	#Get best distribution
	best_fit_name = sorted(distribution_dict, key=lambda x: distribution_dict[x]["mle"])[0]
	logger.debug("Best fitting distribution was: {0}".format(best_fit_name))
	parameters = distribution_dict[best_fit_name]["params"]
	best_distribution = distribution_dict[best_fit_name]["distribution"]

	#Get threshold
	thresholds = best_distribution(*parameters).ppf([percent, 1-percent])
	
	if which == "upper":
		final = thresholds[-1]
	elif which == "lower":
		final = thresholds[0]
	elif which == "both":
		final = tuple(thresholds)

	#Plot fit and threshold
	if plot == True:
		plt.hist(data, bins=30, density=True)

		xlims = plt.xlim()
		ylims = plt.ylim()

		xmin = np.min(data)
		xmax = np.max(data)
		x = np.linspace(xmin, xmax, 100)
		plt.plot(x, best_distribution(*parameters).pdf(x), lw=5, alpha=0.6, label=best_distribution.name)
		
		thresh_list = [final] if not isinstance(final, tuple) else final
		for t in thresh_list:
			plt.axvline(t, color="red")
		plt.legend()
		plt.xlim(xlims)
		plt.ylim(ylims)
		
	
	return(final)

#--------------------------------- Working with TF-COMB objects ---------------------------------#

def is_symmetric(matrix):
	""" Check if a matrix is symmetric around the diagonal """

	if matrix.shape[0] != matrix.shape[1]:
		b = False #not symmetric if matrix is not square
	else:
		b = np.allclose(matrix, matrix.T, equal_nan=True)
	return(b)

def make_symmetric(matrix):
	"""
	Make a numpy matrix matrix symmetric by merging x-y and y-x
	"""
	matrix_T = matrix.T 
	symmetric = matrix + matrix_T

	#don't add up diagonal indices
	di = np.diag_indices(symmetric.shape[0])
	symmetric[di] = matrix_T[di]

	return(symmetric)

def set_contrast(contrast, available_contrasts):
	""" Utility function for the plotting functions of tfcomb.objects.DiffCombObj """

	#Setup contrast to use
	if contrast == None:
		contrast = available_contrasts[0]

	else:
		#Check if contrast is tuple
		if contrast not in available_contrasts:
			raise ValueError("Contrast {0} is not valid (available contrasts are {1})".format(contrast, available_contrasts))

	return(contrast)

# ------------------------- chunk operations ---------------------------------------- #

def analyze_signal_chunks(datasource, threshold):
	""" Evaluating signal for chunks. 
		
		Parameters
		----------
		datasource : pd.DataFrame 
			A (sub-)Dataframe with the (corrected) distance counts for the pairs
		threshold : float
			Threshold for prominence and height in peak calling (see scipy.signal.find_peaks() for detailed information)

		Returns
		--------
		list 
			list of found peaks in form [TF1, TF2, Distance, Peak Heights, Prominences, Prominence Threshold]
		
		See also
		--------
		tfcomb.object.analyze_signal_all
	"""

	# make sure index is correct
	#datasource = datasource.reset_index()
	datasource.index = datasource["TF1"] + "-" + datasource["TF2"]
	distances = datasource.columns.tolist()[2:]
	pairs = zip(datasource["TF1"], datasource["TF2"]) #.index().tolist() #list of pair tuples, e.g. ("NFYA","NFYB")

	# get data column
	distance_cols = np.array([-1 if d == "neg" else d for d in distances]) #neg counts as -1

	#Calculate peaks for each row in datasource
	results = []
	for pair in pairs:

		# get pair
		tf1, tf2 = pair
		ind = "-".join(pair)

		signal = datasource.loc[ind, distances].values
		x = [0] + list(signal) + [0]
		# signal.find_peaks() will not find peaks on first and last position without having 
		# an other number left and right. 

		#Find positions of peaks
		peaks_idx, properties = find_peaks(x, prominence=threshold, height=threshold)

		# subtract the position added above (first zero) 
		peaks_idx = peaks_idx - 1 

		#Get distances from columns
		peak_distances = [distance_cols[idx] for idx in peaks_idx]

		n_peaks = len(peak_distances)
		properties["TF1"] = [tf1]*n_peaks # insert tf1,tf2 names number of peaks times
		properties["TF2"] = [tf2]*n_peaks

		properties["Distance"] = peak_distances
		properties["Threshold"] = threshold

		results.append(properties)
		
	return results

def evaluate_noise_chunks(signals, peaks, method="median", height_multiplier=0.75):
	""" 
	Evaluate the noisiness of a signal for chunks (a chunk can also be the whole dataset). 

	Parameters
	---------
	pairs : list(tuples(str,str))
		list of pairs to perform analysis on 
	signals : pd.Dataframe 
		A (sub-)Dataframe containing signal data for pairs
	method : str, otional
		Method used to get noise measurement, either "median" or "min_max" allowed.
		Default: "median" 
	height_multiplier : float, optional
		Height multiplier (percentage) to calculate cut points. Must be between 0 and 1.
		Default: 0.75
	
	Raises
	------
	ValueError
		If no signal data is given for a pair
	
	Note
	-----
	Constraint: DataFrame with signals need to contain a signal for each pair given within pairs.

	"""
	# make sure index is correct
	signals.index = signals["TF1"] + "-" + signals["TF2"]
	pairs = zip(signals["TF1"], signals["TF2"])
	
	check_value(height_multiplier, vmin=0, vmax=1)

	# get data for each pair
	results = []
	for pair in pairs:
		
		tf1, tf2 = pair
		ind = "-".join(pair)

		#Read information for pair
		signal = signals.loc[ind].iloc[2:] #get signal for pair
		peaks_pair = peaks[(peaks.TF1 == tf1) & (peaks.TF2 == tf2)] # get peaks for specific pair

		results.append([tf1, tf2, _get_noise_measure(peaks_pair, signal, method, height_multiplier)])
		
	return results

def _get_noise_measure(peaks, signal, method, height_multiplier):
	#check method input
	check_string(method, ["median", "min_max"], "method")

	# get the cutting points fot the signal
	cuts = _get_cut_points(peaks, height_multiplier, signal)

	# cut all peaks out of the signal
	for cut in cuts:
		signal.iloc[cut[0]:cut[1]] = np.nan

	measure = None
	if method == "median":
		measure = pd.Series(signal).median()
	elif method == "min_max":
		measure = signal.max() - signal.min()
	
	return float(measure)

def _get_cut_points(peaks, height_multiplier, signal):
	cuts =[]
	for idx,row in peaks.iterrows():
		# get the peak distance
		peak = row.Distance - int(signal.index[0]) # subract min distance for peak offset
		# get the peak height 
		peak_height = signal.iloc[peak]
		# determine cutoff, in common sense this should be "going ~25% down the peak size"
		cut_off = height_multiplier * peak_height
		cuts.append(_expand_peak(peak, cut_off, signal))
	return cuts

def _expand_peak(start_pos, cut_off, signal):
	found_left = False
	found_right = False
	pos_left = start_pos - 1
	pos_right = start_pos + 1

	# expand the peak until both borders are found
	while(not(found_left & found_right)):
		# left side
		if(not found_left):
			# left border not found
			if pos_left <= -1: # check if position less than start of signal
				found_left = True
				left = 0
			elif signal.iloc[pos_left] <= cut_off:
				found_left = True
				left = pos_left  + 1 # we are one to far left
			pos_left -= 1
		
		# right side
		if(not found_right):
			# right border not found	
			if  pos_right == len(signal): # check if position higher than end of signal
				found_right = True
				right = len(signal) - 1
			elif signal.iloc[pos_right] < cut_off:
				found_right = True
				right = pos_right - 1 # we are one to far right
			pos_right += 1
	return(left, right)

def getAllAttr(object, private=False, functions=False):
	"""
	Collect all attributes of an object and return as dict.

	Parameters
	----------
		private : boolean, default False
			If private attributes should be included. Everything with '_' prefix.
		functions : boolean, default False
			If callable attributes ie functions shoudl be included.

	Returns
	----------
		dictionary : 
			Dict of all the objects attributes.
	"""
	output = {}
	
	for attribute_name in dir(object):
		attribute_value = getattr(object, attribute_name)
		
		if not private and attribute_name.startswith("_"):
			continue
		if not functions and callable(attribute_value):
			continue
		
		output[attribute_name] = attribute_value
		
	return output
