import os
import pandas as pd
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import networkx as nx
import graphviz
from adjustText import adjust_text
import copy
import distutils
from distutils import util
import sys

import tfcomb
from tfcomb.utils import check_columns, check_type, check_string, check_value, random_string
from tfcomb.logging import TFcombLogger, InputError
import tobias

# fix 'dot' not found error
# only if conda is found
# https://stackoverflow.com/a/51267131
if os.path.exists(os.path.join(sys.prefix, 'conda-meta')):
	# add install path of active conda bin
	os.environ["PATH"] += os.pathsep + os.path.join(sys.prefix, 'bin')

def bubble(rules_table, yaxis="confidence", size_by="TF1_TF2_support", color_by="lift", figsize=(7,4), save=None):
	""" 
	Plot bubble plot with TF1-TF2 pairs on the x-axis and a choice of measure on the y-axis, as well as color and size of bubbles. 

	Parameters
	----------
	rules_table : pandas.DataFrame
		Dataframe containing data to plot. 
	yaxis : str, optional
		Column containing yaxis information. Default: "confidence".
	size_by : str
		Default: "TF1_TF2_support".
	color_by : str
		Default: None
	figsize : tuple
		Default: (7,7).
	save : str, optional
		Save the plot to the file given in 'save'. Default: None.

	Returns
	--------
	ax
	"""

	check_columns(rules_table, [yaxis, color_by, size_by])	
	check_type(figsize, tuple, "figsize")

	fig, ax = plt.subplots(figsize=figsize)

	with sns.axes_style("whitegrid"):
		
		ax = sns.scatterplot(
							data=rules_table,
							ax=ax,
							x=rules_table.index, 
							y=yaxis, 
							hue=color_by, 
							size=size_by,
							palette="PuBu", 
							edgecolor=".7",	
		)
	
	#Set legend
	sns.move_legend(ax, "center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)

	# Tweak the figure to finalize
	labels = list(rules_table.index)
	ax.set_ylabel(yaxis, fontsize=12)
	ax.set_xlabel("Co-occurring pairs", fontsize=12)

	ax.set_xticks(range(len(labels))) #explicitly set xticks to prevent matplotlib error
	ax.set_xticklabels(labels, rotation=45, ha="right")
	ax.grid(color="0.9") #very light grey
	ax.set_axisbelow(True) #prevent grid from plotting above points
	
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	if save is not None:
		plt.savefig(save, dpi=600, bbox_inches="tight")

	return(ax)


def heatmap(rules_table, columns="TF1", rows="TF2", color_by="cosine", figsize=(7,7), save=None):
	"""
	Plot heatmap with TF1 and TF2 on rows and columns respectively. Heatmap colormap is chosen by .color_by.

	Parameters
	----------
	rules_table : pandas.DataFrame
		The <CombObj>.rules table calculated by market basket analysis
	columns : str, optional
		The name of the column in rules_table to use as heatmap column names. Default: TF1.
	rows : str, optional
		The name of the column in rules_table to use as heatmap row names. Default: TF2.
	color_by : str, optional
		The name of the column in rules_table to use as heatmap colors. Default: "cosine".
	figsize : tuple
		The size of the output heatmap. Default: (7,7)
	save : str
		Save the plot to the file given in 'save'. Default: None.
	"""

	#Test input format
	check_columns(rules_table, [columns, rows, color_by])

	# Create support table for the heatmap
	pivot_table = rules_table.pivot(index=rows, columns=columns, values=color_by)

	#Convert any NaN to null
	pivot_table = pivot_table.fillna(0)

	#Mask any NaN/0 values
	mask = np.zeros_like(pivot_table)
	mask[np.isnan(pivot_table)] = True

	#Choose cmap based on values of 'color_by' columns
	colorby_values = rules_table[color_by]
	if np.min(colorby_values) < 0:
		cmap = "bwr"	#divergent colormap
		center = 0
	else:
		cmap = "PuBu"
		center = None

	row_cluster = True if pivot_table.shape[0] > 1 else False
	col_cluster = True if pivot_table.shape[1] > 1 else False

	#Plot heatmap
	h = sns.clustermap(pivot_table, 
								mask=mask,
								cbar=True, 
								cmap=cmap,
								center=center,
								row_cluster=row_cluster,
								col_cluster=col_cluster,
								cbar_kws={'label': color_by}, 
								xticklabels=True,
								yticklabels=True,
								figsize=figsize
					)

	xticklabels = h.ax_heatmap.axes.get_xticklabels()
	yticklabels = h.ax_heatmap.axes.get_yticklabels()

	h.ax_heatmap.axes.set_xticklabels(xticklabels, rotation=45, ha="right")
	h.ax_heatmap.axes.set_yticklabels(yticklabels, rotation=0)
	h.ax_heatmap.axes.set_facecolor('lightgrey') #color of NA-values

	#plt.title("Top {0} association rules".format(n_rules))
	#plt.tight_layout()

	if save is not None:
		plt.savefig(save, dpi=600)

	return(h)

def scatter(table, x, y, 
				   x_threshold=None, 
				   y_threshold=None, 
				   label=None, 
				   label_fontsize=9, 
				   label_color="red",
				   title=None,
				   save=None,
				   **kwargs):
	"""
	Plot scatter-plot of x/y values within table. Can also set thresholds and label values within plot.

	Parameters
	-----------
	table : pd.DataFrame
		A table containing columns of 'measure' and 'pvalue'.
	x : str
		Name of column in table containing values to map on the x-axis.
	y : str
		Name of column in table containing values to map on the y-axis.
	x_threshold : float, tuple of floats or None, optional
		Gives the option to visualize an x-axis threshold within plot. If None, no measure threshold is set. Default: None.
	y_threshold : float, tuple of floats or None, optional
		Gives the option to visualize an y-axis threshold within plot. If None, no measure threshold is set. Default: None.
	label : str or list, optional
		If None, no point labels are plotted. If "selection", the . Default: None.
	label_fontsize : float, optional
		Size of labels. Default: 9.
	label_color : str, optional
		Color of labels. Default: 'red'.
	title : str, optional
		Title of plot. Default: None.
	kwargs : arguments
		Any additional arguments are passed to sns.jointplot.
	"""

	check_columns(table, [x, y])
	
	#Handle thresholds being either float or tuple
	if x_threshold is not None:
		x_threshold = (x_threshold,) if not isinstance(x_threshold, tuple) else x_threshold
		for threshold in x_threshold:
			check_value(threshold, name="x_threshold")

	if y_threshold is not None:
		y_threshold = (y_threshold,) if not isinstance(y_threshold, tuple) else y_threshold
		for threshold in y_threshold:
			check_value(threshold, name="y_threshold")

	#Plot all data
	x_finite = table[x][~np.isinf(table[x].astype(float))]
	y_finite = table[y][~np.isinf(table[y].astype(float))]

	g = sns.jointplot(x=x_finite, y=y_finite, space=0, **kwargs) #, joint_kws={"s": 100})

	#Plot thresholds
	if x_threshold is not None:
		for threshold in x_threshold:
				g.ax_joint.axvline(threshold, linestyle="--", color="grey")
				g.ax_marg_x.axvline(threshold, linestyle="--", color="grey")

	if y_threshold is not None:
		for threshold in y_threshold:
			g.ax_joint.axhline(threshold, linestyle="--", color="grey")
			g.ax_marg_y.axhline(threshold, linestyle="--", color="grey")

	## Mark selection of pairs below above thresholds in red
	if x_threshold is not None or y_threshold is not None:
		if x_threshold is not None:
			if len(x_threshold) == 1: 
				x_threshold = (-np.inf, x_threshold[0])  #assume that value is lower bound

		if y_threshold is not None:
			if len(y_threshold) == 1:
				y_threshold = (-np.inf, y_threshold[0]) 

		#Set threshold to minimum if not set
		selection = table[((table[x] <= x_threshold[0]) | (table[x] >= x_threshold[1])) &
						 ((table[y] <= y_threshold[0]) | (table[y] >= y_threshold[1]))]
		n_selected = len(selection) #including any non-finite values
		
		#Mark chosen TF pairs in red
		xvals = selection[x]
		xvals_finite = xvals[~np.isinf(xvals)]
		yvals = selection[y]
		yvals_finite = yvals[~np.isinf(yvals)]
		_ = sns.scatterplot(x=xvals_finite, y=yvals_finite, ax=g.ax_joint, color="red", linewidth=0, 
							label="Selection (n={0})".format(n_selected))

	#Label given indices
	if label is not None:
		if isinstance(label, list):

			#Check if labels are within table index
			selection = table.loc[label, :]
			txts = _add_labels(selection, x, y, g.ax_joint, color=label_color, label_fontsize=label_fontsize)

		elif label == "selection":
			txts = _add_labels(selection, x, y, g.ax_joint, color=label_color, label_fontsize=label_fontsize)

		elif label == "all":
			txts = _add_labels(table, x, y, g.ax_joint, color=label_color, label_fontsize=label_fontsize)

		#Adjust positions of labels
		adjust_text(txts, 
						x=table[x].tolist(), 
						y=table[y].tolist(), 
						ax=g.ax_joint,
						#add_objects=[], 
						text_from_points=True, 
						arrowprops=dict(arrowstyle='-', color='black', lw=0.5),
						expand_points=(1.2, 1.2),
						expand_text=(1.2, 1.2)
						)

	if title is not None:
		g.ax_marg_x.set_title(title)

	#Save plot to file
	if save is not None:
		plt.savefig(save, dpi=600, bbox_inches="tight")

	return(g)

#Add labels to ax
def _add_labels(table, x, y, ax, color="black", label_col=None, label_fontsize=9):
	""" Utility to add labels to coordinates 

	Parameters
	----------
	table : pandas.DataFrame
		A dataframe containing coordinates and labels to plot.
	x : str
		The name of a column in table containing x coordinates.
	y : str
		The name of a column in table containing y cooordinates.
	ax : plt axes
		Axes to plot texts on.
	color : str, optional
		Color of label text. Default: "black".
	label_col : str, optional
		Name of column containing labels to plot. Default: None (label is table index)
	label_fontsize : str, optional
		Size of labels. Default: 9.

	Returns 
	--------
	None 
		The labels are added to ax in place
	"""

	#Check if columns are in table
	tfcomb.utils.check_columns(table, [x,y,label_col]) #label_col is not checked if it is None

	#Add texts
	txts = []
	for label, row in table.iterrows():
		coord = (row[x], row[y])

		if label_col != None:
			label = row[label_col]

		txts.append(ax.text(coord[0], coord[1], label, fontsize=label_fontsize, color=color))

	return(txts)


def go_bubble(table, aspect="MF", n_terms=20, threshold=0.05, save=None):
	"""
	Plot a bubble-style plot of GO-enrichment results.

	Parameters
	--------------
	table : pandas.DataFrame 
		The output of tfcomb.analysis.go_enrichment.
	aspect : str
		One of ["MF", "BP", "CC"]
	n_terms : int
		Maximum number of terms to show in graph. Default: 20
	threshold : float between 0-1
		The p-value-threshold to show in plot.
	save : str, optional
		Save the plot to the file given in 'save'. Default: None.
	
	Returns
	----------
	ax

	"""
	
	check_string(aspect, ["BP", "CC", "MF"], "aspect")
	#aspect has to be one of {'BP', 'CC', 'MF'}
	
	#Choose aspect
	aspect_table = table[table["NS"] == aspect]
	aspect_table.loc[:,"-log(p-value)"] = -np.log(aspect_table["p_fdr_bh"])
	aspect_table.loc[:,"n_genes"] = aspect_table["study_count"]

	#Sort by pvalue and ngenes
	aspect_table = aspect_table.sort_values(["-log(p-value)", "p_uncorrected"], ascending=False)
	aspect_table = aspect_table.iloc[:n_terms,:] #first n rows

	#Plot enriched terms 
	#todo: size of plot depending on number of terms to show
	ax = sns.scatterplot(x="-log(p-value)", 
								y="name",
								size="n_genes",
								#sizes=(20,500),
								#alpha=0.5,
								hue="-log(p-value)",
								data=aspect_table, 
								#ax=this_ax
								)

	ax.set_title(aspect) #, pad=20, size=15)		
	ax.axvline(-np.log(threshold), color="red")
	ax.set_ylabel(aspect)
	ax.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
	ax.grid()

	if save is not None:
		plt.savefig(save, dpi=600)

	return(ax)

def _truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
	""" 
	Create a colormap with only a subset of the original range.
	
	Source: https://stackoverflow.com/a/18926541
	"""
	new_cmap = colors.LinearSegmentedColormap.from_list(
		'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
		cmap(np.linspace(minval, maxval, n)))

	new_cmap.set_bad
	return new_cmap

def _rgb_to_hex(rgb):
	return '#%02x%02x%02x' % rgb

def _values_to_cmap(values, plt_cmap=None):
	""" Map values onto a cmap function taking value and returning hex color.
	
	Parameters
	------------
	values : list-like object
		An object containing values to be mapped to colors.
	plt_cmap : str, optional
		Name of a matplotlib colormap to use. Default: None (colors are automatically chosen).
	"""

	#Decide which colormap to use
	colormap_binary = colors.ListedColormap(['lightblue', 'blue'])
	colormap_red = _truncate_colormap(plt.cm.Reds, minval=0.3, maxval=0.7)
	colormap_blue = _truncate_colormap(plt.cm.Blues_r, minval=0.3, maxval=0.7)
	colormap_divergent = _truncate_colormap(plt.cm.bwr, minval=0.1, maxval=0.9)
	colormap_discrete = _truncate_colormap(plt.cm.jet, minval=0.3, maxval=0.7)
	colormap_custom = copy.copy(matplotlib.cm.get_cmap(plt_cmap))
	
	#First, convert values to bool if possible
	values = _convert_boolean(values)

	#Check if values are strings 
	if sum([isinstance(s, str) for s in values]) > 0: #values are strings, cmap should be discrete
		cmap = colormap_discrete if plt_cmap is None else colormap_custom
		cmap.set_bad(color="grey")

		values_unique = list(set(values))
		floats = np.linspace(0,1,len(values_unique))
		name2val = dict(zip(values_unique, floats)) #map strings to cmap values

		color_func = lambda string: _rgb_to_hex(cmap(name2val[string], bytes=True)[:3])
		typ = "string"

	#Check if values are boolean
	elif sum([isinstance(s, bool) for s in values]) > 0:

		cmap = colormap_binary if plt_cmap is None else colormap_custom
		cmap.set_bad(color="grey") #color for NaN

		color_func = lambda value: _rgb_to_hex(cmap(int(value), bytes=True)[:3])
		typ = "bool"
	
	#Values are int/float
	else:

		#Check if values contain NaN
		clean_values = np.array(values)[~np.isnan(values)]

		#Get min and max
		vmin, vmax = np.min(clean_values), np.max(clean_values)

		if plt_cmap != None: #plt_cmap is given explicitly
			cmap = colormap_custom
		elif vmin >= 0 and vmax >= 0:
			cmap = colormap_red 
		elif vmin < 0 and vmax <= 0:
			cmap = colormap_blue
		elif vmin < 0 and vmax >= 0:
			cmap = colormap_divergent
			max_abs = max([abs(vmin), abs(vmax)])
			vmin = -max_abs #make sure that convergent maps are centered at 0
			vmax = max_abs
			
		#Normalize values and create cmap
		norm_func = plt.Normalize(vmin=vmin, vmax=vmax)
		sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_func)
		cmap = sm.get_cmap()
		cmap.set_bad(color="grey") #set color for np.nan
		color_func = lambda value: _rgb_to_hex(cmap(norm_func(value), bytes=True)[:3])
		typ = "continuous"

	return((typ, color_func))

def _isnan(num):
	return num != num

def _convert_boolean(values):
	""" Converts a list of boolean/string/nan values into boolean values - but only if all values could be converted """

	#Convert any boolean values
	bool_vals = ["y", "yes", "t", "true", "on", "1", "n", "no", "f", "false", "off", "0"]
	converted = [bool(distutils.util.strtobool(val)) if (isinstance(val, str) and (val.lower() in bool_vals)) else val for val in values]

	#Check if clean values contain only bool
	clean = [val for val in converted if _isnan(val) == False]
	n_bool = sum([isinstance(val, bool) for val in clean])
	
	if n_bool == len(clean):
		return(converted) #all values could be converted - these are boolean values
	else:
		return(values)     #not all values could be converted - these are not boolean values


def _get_html_colormap(colormap, vmin, vmax, n):
	""" Function to create a text-colormap for network legend """

	html = ""
	steps = np.linspace(vmin, vmax, n)
	for step in steps:
		color = colormap(step)
		html += f'<FONT COLOR="{color}">█</FONT>'

	return(html)


def network(network, 
				color_node_by=None,
				color_edge_by=None,
				size_node_by=None, 
				size_edge_by=None,
				engine="sfdp", 
				size="8,8", 
				min_edge_size=2,
				max_edge_size=8,
				min_node_size=14,
				max_node_size=20,
				legend_size='auto',
				node_border=False,
				node_cmap=None,
				edge_cmap=None,
				node_attributes={},
				save=None,
				verbosity=1,
				):
	"""
	Plot network of a networkx object using Graphviz for python.

	Parameters
	-----------
	network : networkx.Graph
		A networkx Graph/DiGraph object containing the network to plot.
	color_node_by : str, optional
		The name of a node attribute
	color_edge_by : str, optional
		The name of an edge attribute
	size_node_by : str, optional
		The name of a node attribute
	size_edge_by : str, optional
		The name of an edge attribute
	engine : str, optional
		The graphviz engine to use for finding network layout. Default: "sfdp".
	size : str, optional
		Size of the output figure. Default: "8,8".
	min_edge_size : float, optional
		Default: 2. 
	max_edge_size : float, optional
		Default: 8.
	min_node_size : float, optional
		Default: 14.
	max_node_size : float, optional
		Default: 20.
	legend_size : int, optional
		Fontsize for legend explaining color_node_by/color_edge_by/size_node_by/size_edge_by. Set to 0 to hide legend. Default: 'auto'.
	node_border : bool, optional
		Whether to plot border on nodes. Can be useful if the node colors are very light. Default: False.
	node_cmap : str, optional
		Name of colormap for node coloring. Default: None (colors are automatically chosen).
	edge_cmap : str, optional
		Name of colormap for edge coloring. Default: None (colors are automatically chosen).
	node_attributes : dict, optional
		Additional node attributes to apply to graph. Default: No additional attributes.
	save : str, optional
		Path to save network figure to. Format is inferred from the filename - if not valid, the default format is '.pdf'.	
	verbosity : int, optional
		verbosity of the logging. Default: 1.

	Raises
	-------
	TypeError
		If network is not a networkx.Graph object
	InputError 
		If any of 'color_node_by', 'color_edge_by' or 'size_edge_by' is not in node/edge attributes, or if 'engine' is not a valid graphviz engine.
	"""
		
	# Setup logger
	logger = TFcombLogger(verbosity)
	
	############### Test input ###############
	
	check_type(network, [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
	if legend_size is not 'auto':
		check_value(legend_size, vmin=0, integer=True, name="legend_size")

	#Read nodes and edges from graph and check attributes
	node_view = network.nodes(data=True)
	edge_view = network.edges(data=True)

	node_attributes_list = list(list(node_view)[0][-1].keys())    
	edge_attributes_list = list(list(edge_view)[0][-1].keys())

	for att in [color_node_by, size_node_by]:
		if (att is not None) and (att not in node_attributes_list):
			raise InputError("Attribute '{0}' is not available in the network node attributes. Available attributes are: {1}".format(att, node_attributes_list))

	for att in [color_edge_by, size_edge_by]:
		if (att is not None) and (att not in edge_attributes_list):
			raise InputError("Attribute '{0}' is not available in the network edge attributes. Available attributes are: {1}".format(att, edge_attributes_list))

	#Check if engine is within graphviz
	if engine not in graphviz.ENGINES:
		raise ValueError("The given engine '{0}' is not in graphviz available engines: {1}".format(engine, graphviz.ENGINES))

	# Check number of edges
	if len(network.edges) > 10000:
		logger.warning(f"Detected more than 10.000 edges ({len(network.edges)}). This can result in issues when using jupyter.")
	
	#todo: check size with re
	
	############### Initialize graph ###############

	#Establish if network is directional
	if not nx.is_directed(network):
		dot = graphviz.Graph(engine=engine)
	else:
		dot = graphviz.Digraph(engine=engine)

	dot.attr(size=size)
	dot.attr(outputorder="edgesfirst")
	dot.attr(overlap="false")

	############ Setup colormaps/sizemaps ############
	map_value = {}
	map_type = {}
	
	#Node color
	if color_node_by != None:
		all_values = [node[-1][color_node_by] for node in node_view]
		typ, cmap = _values_to_cmap(all_values, node_cmap)
		map_type["node_color"] = typ
		map_value["node_color"] = cmap
	
	#Edge color
	if color_edge_by != None:
		all_values = [edge[-1][color_edge_by] for edge in edge_view]
		typ, cmap = _values_to_cmap(all_values, edge_cmap)
		map_type["edge_color"] = typ
		map_value["edge_color"] = cmap
	
	#Node size
	if size_node_by != None: #must be continuous
		all_values = [node[-1][size_node_by] for node in node_view]
		nmin, nmax = np.min(all_values), np.max(all_values)
		map_value["node_size"] = lambda value: np.round((value-nmin)/(nmax-nmin)*(max_node_size-min_node_size)+min_node_size, 2)
	
	#Edge size
	if size_edge_by != None: #must be continuous
		all_values = [edge[-1][size_edge_by] for edge in edge_view]
		vmin, vmax = np.min(all_values), np.max(all_values)
		map_value["edge_size"] = lambda value: np.round((value-vmin)/(vmax-vmin)*(max_edge_size-min_edge_size)+min_edge_size, 2)

	
	############### Add nodes to network ##############

	logger.debug("Adding nodes to dot network")
	for node in node_view:
	
		node_name = node[0]
		node_att = node[1]
		
		attributes = {} #attributes for dot
		attributes["style"] = "filled"
		attributes["width"] = "0.25" #minimum width; will expand to fit label
		attributes["height"] = "0.25" #minimum width; will expand to fit label
		attributes["fixedsize"] = "false" #automatically adjust node sizes
		attributes["fillcolor"] = "lightgrey"

		#Set color of node border (default: black)
		if node_border == False:
			attributes["color"] = "none"

		#Set node color
		if color_node_by != None:
			value = node_att[color_node_by]
			attributes["fillcolor"] = map_value["node_color"](value)

			#Adjust label color based on darkness of fill
			R, G, B = matplotlib.colors.to_rgb(attributes["fillcolor"]) #from hex to rgb
			luminance = (0.2126*R + 0.7152*G + 0.0722*B)
			if luminance < 0.5: #if fill is dark, the font should be white
				attributes["fontcolor"] = "white"
			
		#Set node size
		if size_node_by != None:
			value = node_att[size_node_by]
			attributes["fontsize"] = str(map_value["node_size"](value))

		#Apply any additional attributes
		for key in node_attributes:
			attributes[key] = str(node_attributes[key])

		#After collecting all attributes; add node with attribute dict
		logger.spam("Adding node {0}".format(node_name))
		dot.node(node_name, _attributes=attributes)

	############### Add edges to network ###############
	logger.debug("Adding edges to dot network")

	for edge in edge_view:
	
		node1, node2 = edge[:2]
		edge_att = edge[-1]
		
		attributes = {}
		attributes["penwidth"] = str(min_edge_size) #default size; can be overwritten by size_edge_by
		
		#Set edge color
		if color_edge_by != None:
			value = edge_att[color_edge_by]
			attributes["color"] = map_value["edge_color"](value)
		
		#Set edge size
		if size_edge_by != None:
			value = edge_att[size_edge_by]
			attributes["penwidth"] = str(map_value["edge_size"](value))

		#After collecting all edge attributes; add edge to dot object
		dot.edge(node1, node2, _attributes=attributes)

	#Plot legend to dot object
	if legend_size == 'auto':
		n_nodes = len(node_view)
		legend_size = int(10 + n_nodes*0.1) #incrementally increasing size 
		logger.debug("legend_size is estimated at: {0}".format(legend_size))

	if legend_size > 0:
		h = int(legend_size/3)
		spacer = f'<TR><TD HEIGHT="{h}"></TD></TR>' #spacer between rows
		
		#Start building legend
		html_legend = f'<<FONT POINT-SIZE="{legend_size}" FACE="ARIAL">'
		#TODO: add a bit of space between nodes and legend position
		html_legend += '<TABLE ALIGN="LEFT" BORDER="1" CELLBORDER="0" CELLSPACING="0" VALIGN="MIDDLE">'
		html_legend += spacer

		if color_node_by is not None:
			html_legend += f'<TR><TD ALIGN="LEFT" > <b>Nodes colored by:</b> </TD><TD ALIGN="LEFT">  {color_node_by}  </TD>'
			
			#Whether to create colormap
			if map_type["node_color"] == "continuous":
				all_values = [node[-1][color_node_by] for node in node_view]
				min_val = round(np.min(all_values), 2)
				max_val = round(np.max(all_values), 2)
				html_colormap = _get_html_colormap(map_value["node_color"], min_val, max_val, 10)
				html_legend += f'<TD ALIGN="RIGHT"><i>{min_val}</i> </TD><TD>{html_colormap}</TD><TD ALIGN="left"><i>{max_val}</i> </TD>'
			html_legend += '</TR>' + spacer 

		if color_edge_by is not None:
			html_legend += f'<TR><TD ALIGN="LEFT"> <b>Edges colored by:</b> </TD><TD ALIGN="LEFT">  {color_edge_by}  </TD>'

			#Whether to create colormap
			if map_type["edge_color"] == "continuous":
				all_values = [edge[-1][color_edge_by] for edge in edge_view]
				min_val = round(np.min(all_values), 2)
				max_val = round(np.max(all_values), 2)
				html_colormap = _get_html_colormap(map_value["edge_color"], min_val, max_val, 10)
				html_legend += f'<TD ALIGN="RIGHT"><i>{min_val}</i> </TD><TD>{html_colormap}</TD><TD ALIGN="left"><i>{max_val}</i> </TD>'
			html_legend += "</TR>" + spacer

		if size_node_by is not None: 
			html_legend += f'<TR><TD ALIGN="LEFT"> <b>Nodes sized by:</b> </TD><TD ALIGN="LEFT">  {size_node_by}  </TD>'

			all_values = [node[-1][size_node_by] for node in node_view]
			min_val = round(np.min(all_values), 2)
			max_val = round(np.max(all_values), 2)
			html_legend += f'<TD></TD><TD ALIGN="CENTER"> <i>{min_val} </i> ● ⬤<i> {max_val}</i> </TD><TD></TD>'
			html_legend += '</TR>' + spacer

		if size_edge_by is not None:
			html_legend += f'<TR><TD ALIGN="LEFT"> <b>Edges sized by:</b> </TD><TD ALIGN="LEFT">  {size_edge_by}  </TD>'

			all_values = [edge[-1][size_edge_by] for edge in edge_view]
			min_val = round(np.min(all_values), 2)
			max_val = round(np.max(all_values), 2)
			html_legend += f'<TD></TD><TD ALIGN="CENTER"> <i>{min_val}</i>  ◄ <i>{max_val}</i> </TD><TD></TD>'
			html_legend += '</TR>' + spacer

		#Finalize legend
		html_legend += '</TABLE>'
		html_legend += '</FONT>>'

		#Add legend and location to dot obj
		dot.attr(label=html_legend)
		dot.attr(labelloc="b")
		dot.attr(labeljust="r")

	############### Save to file ###############
	if save != None:

		#Set dpi for output render (not for visualized, as this doesn't work with notebook)
		dot_render = copy.deepcopy(dot)
		#dot_render.attr(dpi="600")

		splt = os.path.splitext(save)
		file_prefix = "".join(splt[:-1])
		fmt = splt[-1].replace(".", "")

		if fmt != ".pdf":
			dot_render.attr(dpi="600") #for .png's to ensure quality
		
		if fmt not in graphviz.FORMATS:
			logger.warning("File ending .{0} is not supported by graphviz/dot. Network will be saved as .pdf.".format(fmt))
			fmt = "pdf"
		dot_render.render(filename=file_prefix, format=fmt, cleanup=True)
	
	return(dot)


def genome_view(TFBS, 
					window_chrom=None,
					window_start=None, 
					window_end=None,
					window=None,
					fasta=None,
					bigwigs=None,
					bigwigs_sharey=False,
					TFBS_track_height=4,
					title=None,
					highlight=None,
					save=None,
					verbosity=1):

	""" Plot TFBS in genome view via the 'DnaFeaturesViewer' package. 
	
	Parameters
	--------------
	TFBS : list
		A list of OneTFBS objects or any other object containing .chrom, .start, .end and .name variables.
	window_chrom : str, optional if 'window' is given
		The chromosome of the window to show. 
	window_start : int, optional if 'window' is given
		The genomic coordinates for the start of the window.
	window_end : int, optional if 'window' is given
		The genomic coordinates for the end of the window.
	window : Object with .chr, .start, .end 
		If window_chrom/window_start/window_end are not given, window can be given as an object containing .chrom, .start, .end variables
	fasta : str, optional
		The path to a fasta file containing sequence information to show. Default: None.
	bigwigs : str, list or dict of strings, optional
		Give the paths to bigwig signals to show within graph. Default: None.
	bigwigs_sharey : bool or list, optional
		Whether bigwig signals should share y-axis range. If True, all signals will be shared. 
		It is also possible to give a list of bigwig indices (starting at 0), which should share y-axis values, e.g. [0,1,3] for the 1st, 2nd and 4th bigwig to share signal.
		If list of lists, each lists correspond to a grouping, e.g. [[0,2], [1,3]]. Default: False.
	TFBS_track_height : float, optional
		Relative track height of TFBS. Default: 4.
	title : str, optional
		Title of plot. Default: None.
	highlight : list, optional
		A list of OneTFBS objects or any other object containing .chrom, .start, .end and .name variables.
	save : str, optional
		Save the plot to the file given in 'save'. Default: None.
	"""

	logger = TFcombLogger(verbosity)

	#Test if package is available
	if tfcomb.utils.check_module("dna_features_viewer") == True:
		from dna_features_viewer import GraphicFeature, GraphicRecord

	#----------------- Format input data ----------------#
	
	#Establish which region to show
	logger.debug("Subsetting TFBS to window")

	if window_chrom != None and window_start != None and window_end != None:
		window = tfcomb.utils.OneTFBS([window_chrom, window_start, window_end])

	#Subset on windows or take all TFBS?
	if window != None:
		TFBS = [site for site in TFBS if (site.chrom == window.chrom) and (site.start >= window.start) and (site.end <= window.end)]

	else: #show all TFBS
		logger.warning("No window was set - showing the first 100 sites in .TFBS")

		#Only keep first chromosome of TFBS
		chrom = TFBS[0].chrom
		TFBS = [site for site in TFBS if site.chrom == chrom]

		#Set max amount of TFBS to show
		TFBS = TFBS[:100]

		#Get min/max of all sites
		window_start = np.inf
		window_end = -np.inf
		for site in TFBS:
			
			if site.end > window_end:
				window_end = site.end
			if site.start < window_start:
				window_start = site.start

		window = tobias.utils.regions.OneRegion([chrom, window_start, window_end])

	logger.debug(window)
		
	window_length = window.end - window.start

	#Establish how many bigwig paths were given
	bigwigs = [] if bigwigs == None else bigwigs
	bigwigs = [bigwigs] if isinstance(bigwigs, str) else bigwigs
	n_bigwig_tracks = len(bigwigs)

	#How many subplots to create
	n_tracks = 1 + n_bigwig_tracks

	#------------ Create plt subplots ------------#

	height_ratios = [TFBS_track_height] + [1]*n_bigwig_tracks

	fig, axes = plt.subplots(n_tracks, 1, 
								sharex=True, 
								figsize=(8,TFBS_track_height+n_bigwig_tracks), 
								constrained_layout=True,
								gridspec_kw={"height_ratios": height_ratios}
								)
	axes = [axes] if not isinstance(axes, np.ndarray) else axes #for n_tracks == 1

	#------------ Add TFBS features to plot ------------#

	## Add features from TFBS list
	features = []
	colors_used = {}

	if len(TFBS) == 0:
		logger.warning("No TFBS to show within the given window.")

	for site in TFBS:

		strand_convert = {"+":1, "-":-1}
		label = site.name
		#Get unique color for this TFBS
		
		#Add feature
		feature = GraphicFeature(start=site.start, end=site.end, strand=strand_convert.get(site.strand, None), label=label)
		features.append(feature)

	#Add sequence track
	if fasta is not None:

		#Pull sequence from fasta file
		genome_obj = tfcomb.utils.open_genome(fasta)
		sequence = genome_obj.fetch(window.chrom, window.start, window.end)

	else:
		sequence = None

	record = GraphicRecord(first_index=window.start, 
						   sequence_length=window_length, 
						   sequence=sequence,
						   features=features, 
						   labels_spacing=20
						   )
	with_ruler = True if n_bigwig_tracks == 0 else False
	record.plot(ax=axes[0], with_ruler=with_ruler)
	plt.xticks(rotation=45, ha="right", color="grey")

	#Plot sequence
	if sequence is not None:
		record.plot_sequence(axes[0], y_offset=1)

	#------------ Add additional features -----------#
	#Add bigwig track(s)
	if bigwigs is not None:
		for i, bigwig_f in enumerate(bigwigs):

			#Open pybw and pull values
			pybw = tfcomb.utils.open_bigwig(bigwig_f)
			signal = tobias.utils.regions.OneRegion.get_signal(window, pybw)

			#Add signal to plot
			bigwig_name = os.path.splitext(os.path.basename(bigwig_f))[0]
			xvals = np.arange(window.start, window.end)
			axes[i+1].fill_between(xvals, signal, step="mid")
			#axes[i+1].step(xvals, signal, where="mid")
			axes[i+1].set_ylabel(bigwig_name, rotation=0, ha="right")
			axes[i+1].yaxis.tick_right()

			ymin = np.min(signal)
			ymax = np.max(signal)
			pad = (ymax - ymin)*0.2
			axes[i+1].set_ylim(ymin-pad, ymax+pad)
			

			#Set spine color to grey
			axes[i+1].tick_params(color='grey', labelcolor='grey')
			#plt.setp(axes[i+1].spines.values(), color="grey")
			plt.setp([axes[i+1].get_xticklines(), axes[i+1].get_yticklines()], color="grey")

		#Whether to share y across all bigwig tracks
		if bigwigs_sharey != False:

			#Establish which groups should share y-axis
			if bigwigs_sharey == True: #share y across all bigwig tracks
				grouping = [list(range(len(bigwigs)))]
			elif not isinstance(bigwigs_sharey[0], list):
				grouping = [bigwigs_sharey] #list of lists - only one group
			else:
				grouping = bigwigs_sharey #already list of lists

			#Set ylim across groups
			for group in grouping:
				group_ylims = list(zip(*[axes[i+1].get_ylim() for i in group]))

				ymin = min(group_ylims[0])
				ymax = max(group_ylims[1])

				for i in group:
					axes[i+1].set_ylim(ymin, ymax)

		#Highlight sites given
		if highlight != None:

			#Get sites within the window
			highlight_sites = []
			for site in highlight:
				if site.chrom == window.chrom:
					if max([site.start, site.end]) > window.start and min([site.start, site.end]) < window.end:
						highlight_sites.append(site)

			#Plot highlight
			xlim = axes[0].get_xlim()
			for i in range(len(bigwigs)):
				pass
				#axes[i+1].

	if title is not None:
		axes[0].set_title(title, y=1.05)

	plt.xlabel(window.chrom, color="grey")

	#-------------- Done with plot; show/save -----------#

	if save is not None:
		plt.savefig(save, dpi=600)

	return(axes)
	
