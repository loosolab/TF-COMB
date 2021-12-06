import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import networkx as nx
import graphviz
from adjustText import adjust_text
import copy
import distutils
from distutils import util


from tfcomb.utils import check_columns, check_type, check_string, check_value
from tfcomb.logging import TFcombLogger


def bubble(rules_table, yaxis="confidence", size_by="TF1_TF2_support", color_by="lift", figsize=(7,4), save=None):
	""" 
	Plot bubble plot with TF1-TF2 pairs on the x-axis and a choice of measure on the y-axis, as well as color and size of bubbles. 

	Parameters
	----------
	rules_table : pandas.DataFrame
		Dataframe containing 
	yaxis : str, optional
		Default: "confidence".
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
	ax.grid()
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

	# Tweak the figure to finalize
	labels = list(rules_table.index)
	ax.set(ylabel=yaxis, xlabel="Co-occurring pairs")
	ax.set_xticks(range(len(labels))) #explicitly set xticks to prevent matplotlib error
	ax.set_xticklabels(labels, rotation=45, ha="right")

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
				   save=None):
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
	x_finite = table[x][~np.isinf(table[x])]
	y_finite = table[y][~np.isinf(table[y])]

	g = sns.jointplot(x=x_finite, y=y_finite, space=0, linewidth=0.2) #, joint_kws={"s": 100})

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
		_ = sns.scatterplot(x=xvals_finite, y=yvals_finite, ax=g.ax_joint, color="red", linewidth=0.2, 
							label="Selection (n={0})".format(n_selected))

	#Label given indices
	if label is not None:
		if isinstance(label, list):

			#Check if labels are within table index
			
			pass


		elif label == "selection":
			_add_labels(selection, x, y, "index", g.ax_joint, color="red", label_fontsize=label_fontsize)

	#Save plot to file
	if save is not None:
		plt.savefig(save, dpi=600)

	return(g)

#Add labels to ax
def _add_labels(table, x, y, label, ax, color="black", label_fontsize=9):
	""" Utility to add labels to coordinates 

	Parameters
	----------
	table : pandas.DataFrame
		A dataframe containing coordinates and labels to plot.
	x : str
		The name of a column in table containing x coordinates.
	y : str
		The name of a column in table containing y cooordinates.
	label : str
		Name of column or "index" containing labels to plot.
	ax : plt axes


	Returns 
	--------
	None 
		The labels are added to ax in place
	"""

	txts = []
	for l in label:
		coord = [table.loc[l,measure_col], table.loc[l,log_col]]
		
		ax = g.ax_joint
		ax.scatter(coord[0], coord[1], color="red")
	
		txts.append(ax.text(coord[0], coord[1], l, fontsize=label_fontsize))
	
	#Adjust overlapping labels
	adjust_text(txts, ax=ax, add_objects=[], text_from_points=True, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))  #, expand_text=(0.1,1.2), expand_objects=(0.1,0.1))




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
	""" Map values onto a cmap function taking value and returning hex color """

	#Decide which colormap to use
	colormap_binary = colors.ListedColormap(['lightblue', 'blue'])
	colormap_red = _truncate_colormap(plt.cm.Reds, minval=0.3, maxval=0.7)
	colormap_blue = _truncate_colormap(plt.cm.Blues_r, minval=0.3, maxval=0.7)
	colormap_divergent = _truncate_colormap(plt.cm.bwr, minval=0.1, maxval=0.9)
	colormap_discrete = _truncate_colormap(plt.cm.jet, minval=0.3, maxval=0.7)
	
	#First, convert values to bool if possible
	values = _convert_boolean(values)

	#Check if values are strings 
	if sum([isinstance(s, str) for s in values]) > 0: #values are strings, cmap should be discrete
		cmap = colormap_discrete
		cmap.set_bad(color="grey")

		values_unique = list(set(values))
		floats = np.linspace(0,1,len(values_unique))
		name2val = dict(zip(values_unique, floats)) #map strings to cmap values

		color_func = lambda string: _rgb_to_hex(cmap(name2val[string], bytes=True)[:3])

	#Check if values are boolean
	elif sum([isinstance(s, bool) for s in values]) > 0:

		cmap = colormap_binary
		cmap.set_bad(color="grey") #color for NaN

		color_func = lambda value: _rgb_to_hex(cmap(int(value), bytes=True)[:3])

		#sm = plt.cm.ScalarMappable(cmap=plt_cmap, norm=norm_func)
		#cmap = sm.get_cmap()

	#Values are int/float
	else:

		#Check if values contain NaN
		clean_values = np.array(values)[~np.isnan(values)]

		#Get min and max
		vmin, vmax = np.min(clean_values), np.max(clean_values)

		if plt_cmap != None: #plt_cmap is given explicitly
			pass #todo: check that plt_cmap is a colormap
		elif vmin >= 0 and vmax >= 0:
			plt_cmap = colormap_red
		elif vmin < 0 and vmax <= 0:
			plt_cmap = colormap_blue
		elif vmin < 0 and vmax >= 0:
			plt_cmap = colormap_divergent
			
		#Normalize values and create cmap
		norm_func = plt.Normalize(vmin=vmin, vmax=vmax)
		sm = plt.cm.ScalarMappable(cmap=plt_cmap, norm=norm_func)
		cmap = sm.get_cmap()
		cmap.set_bad(color="grey") #set color for np.nan
		color_func = lambda value: _rgb_to_hex(cmap(norm_func(value), bytes=True)[:3])
	
	return(color_func)

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
				save=None,
				verbosity = 1):
	"""
	Plot network of a networkx object using Graphviz for python.

	Parameters
	-----------
	network : networkx.Graph
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
	save : str, optional
		Path to save network figure to. Format is inferred from the filename - if not valid, the default format is '.pdf'.	
	verbosity : int
		verbosity of the logging. Default: 1.

	Raises
	-------
	TypeError
		If network is not a networkx.Graph object
	ValueError 
		If any of 'color_node_by', 'color_edge_by' or 'size_edge_by' is not in node/edge attributes, or if 'engine' is not a valid graphviz engine.
	"""
		
	# Setup logger
	logger = TFcombLogger(verbosity)
	
	############### Test input ###############
	
	check_type(network, [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])

	#Read nodes and edges from graph and check attributes
	node_view = network.nodes(data=True)
	edge_view = network.edges(data=True)

	node_attributes = list(list(node_view)[0][-1].keys())    
	edge_attributes = list(list(edge_view)[0][-1].keys())

	for att in [color_node_by, size_node_by]:
		if (att is not None) and (att not in node_attributes):
			raise ValueError("Attribute '{0}' is not available in the network node attributes. Available attributes are: {1}".format(att, node_attributes))

	for att in [color_edge_by, size_edge_by]:
		if (att is not None) and (att not in edge_attributes):
			raise ValueError("Attribute '{0}' is not available in the network edge attributes. Available attributes are: {1}".format(att, edge_attributes))

	#Check if engine is within graphviz
	if engine not in graphviz.ENGINES:
		raise ValueError("The given engine '{0}' is not in graphviz available engines: {1}".format(engine, graphviz.ENGINES))
	
	#todo: check size with re
	
	############### Initialize graph ###############
	
	dot = graphviz.Graph(engine=engine)
	dot.attr(size=size)
	dot.attr(outputorder="edgesfirst")
	dot.attr(overlap="false")
	
	############ Setup colormaps/sizemaps ############
	map_value = {}
	
	#Node color
	if color_node_by != None:
		all_values = [node[-1][color_node_by] for node in node_view]
		map_value["node_color"] = _values_to_cmap(all_values)
	
	#Edge color
	if color_edge_by != None:
		all_values = [edge[-1][color_edge_by] for edge in edge_view]
		map_value["edge_color"] = _values_to_cmap(all_values)
	
	#Node size
	if size_node_by != None:
		all_values = [node[-1][size_node_by] for node in node_view]
		nmin, nmax = np.min(all_values), np.max(all_values)
		map_value["node_size"] = lambda value: np.round((value-nmin)/(nmax-nmin)*(max_node_size-min_node_size)+min_node_size, 2)
	
	#Edge size
	if size_edge_by != None:
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

		#Set node color
		if color_node_by != None:
			value = node_att[color_node_by]
			attributes["color"] = map_value["node_color"](value)
			
		#Set node size
		if size_node_by != None:
			value = node_att[size_node_by]
			attributes["fontsize"] = str(map_value["node_size"](value))
		
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

	############### Save to file ###############
	if save != None:

		#Set dpi for output render (not for visualized, as this doesn't work with notebook)
		dot_render = copy.deepcopy(dot)
		dot_render.attr(dpi="600")

		splt = os.path.splitext(save)
		file_prefix = "".join(splt[:-1])
		fmt = splt[-1].replace(".", "")
		
		if fmt not in graphviz.FORMATS:
			logger.warning("File ending .{0} is not supported by graphviz/dot. Network will be saved as .pdf.".format(fmt))
			fmt = "pdf"
		dot_render.render(filename=file_prefix, format=fmt, cleanup=True)
	
	return(dot)
