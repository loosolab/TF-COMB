import os
import nxviz
import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import graphviz

from tfcomb.utils import check_columns


def bubble(rules_table, yaxis="confidence", size_by="TF1_TF2_support", color_by="lift", figsize=(7,7), save=None):
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

	ax.grid()

	labels = list(rules_table.index)

	# Tweak the figure to finalize
	ax.set(ylabel=yaxis, xlabel="Co-occurring pairs")
	ax.set_xticklabels(labels, rotation=45, ha="right")

	return(ax)


def heatmap(rules_table, columns="TF1", rows="TF2", color_by="cosine", figsize=(7,7), save=None):
	"""
	Plot heatmap with TF1 and TF2 on rows and columns respectively. Heatmap colormap is chosen by .color_by.

	Parameters
	----------
	rules_table : pandas.DataFrame

	columns : str, optional

	rows : str, optional

	color_by : str, optional

	figsize : tuple
		. Default: (7,7)

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

	#Plot heatmap
	h = sns.clustermap(pivot_table, 
								mask=mask,
								cbar=True, 
								cmap=cmap,
								center=center,
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

	return(h)

def volcano(table, measure=None, pvalue=None, measure_threshold=None, pvalue_threshold=None):
	"""
	Plot volcano-style plots combining a measure and a pvalue.

	Parameters
	-----------
	table : pd.DataFrame
		A table containing columns of 'measure' and 'pvalue'.
	measure : str
		The measure to show on the x-axis.
	pvalue : str
		The column containing p-values (without any log-transformation).
	measure_threshold : float or list of floats

	pvalue_threshold : float between 0-1
		Default: 0.05.
	"""

	check_columns(table, [measure, pvalue])	

	#Convert pvalue to -log10
	table = table.copy() #ensures that we don't change the table in place
	pval_col = "-log10({0})".format(pvalue)
	table[pval_col] = -np.log10(table[pvalue])

	g = sns.jointplot(data=table, x=measure, y=pval_col, space=0) #, joint_kws={"s": 100})

	#Plot thresholds
	if pvalue_threshold is not None:
		g.ax_joint.axhline(-np.log10(pvalue_threshold), linestyle="--", color="grey")
		g.ax_marg_y.axhline(-np.log10(pvalue_threshold), linestyle="--", color="grey") #y-axis (pvalue)

	if measure_threshold is not None:
		
		#if measure_threshold

		g.ax_joint.axvline(measure_threshold, linestyle="--", color="grey")
		g.ax_marg_x.axvline(measure_threshold, linestyle="--", color="grey")	#x-axis (measure) threshold

	## Create selection of pairs below above thresholds
	if measure_threshold is not None or pvalue_threshold is not None:

		#Set threshold to minimum if not set
		pvalue_threshold = np.min(table[pval_col]) if pvalue_threshold is None else pvalue_threshold
		measure_threshold = np.min(table[measure]) if measure_threshold is None else measure_threshold

		selection = table[(table[measure] >= measure_threshold) & 
						  (table[pvalue] <= pvalue_threshold)]
		n_selected = len(selection)

		#Mark chosen TF pairs in red
		xvals = selection[measure]
		yvals = selection[pval_col]
		_ = sns.scatterplot(x=xvals, y=yvals, ax=g.ax_joint, color="red", 
							label="Selection (n={0})".format(n_selected))

	if save is not None:
		pass

	return(g)
	
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
		""

	Returns
	----------
	ax

	"""
	
	#aspect has to be one of {'BP', 'CC', 'MF'}
	

	#Choose aspect
	aspect_table = table[table["NS"] == aspect]
	aspect_table.loc[:,"-log(p-value)"] = -np.log(aspect_table["p_fdr_bh"])
	aspect_table.loc[:,"n_genes"] = aspect_table["study_count"]

	#Sort by pvalue and ngenes
	aspect_table.sort_values("-log(p-value)", ascending=False, inplace=True)
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

	return(ax)

def _truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
	""" 
	Create a colormap with only a subset of the original range.
	
	Source: https://stackoverflow.com/a/18926541
	"""
	new_cmap = colors.LinearSegmentedColormap.from_list(
		'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
		cmap(np.linspace(minval, maxval, n)))
	return new_cmap

def _rgb_to_hex(rgb):
	return '#%02x%02x%02x' % rgb

def _values_to_cmap(values, plt_cmap=None):
	""" Map values onto a cmap function taking value and returning hex color """

	#Decide which colormap to use
	colormap_red = _truncate_colormap(plt.cm.Reds, minval=0.3, maxval=0.7)
	colormap_blue = _truncate_colormap(plt.cm.Blues_r, minval=0.3, maxval=0.7)
	colormap_divergent = _truncate_colormap(plt.cm.bwr, minval=0.1, maxval=0.9)
	
	vmin, vmax = np.min(values), np.max(values)
	if plt_cmap != None: #plt_cmap is given explicitly
		pass #todo: check that plt_cmap is a colormap
	elif vmin >= 0 and vmax >= 0:
		plt_cmap = colormap_red
	elif vmin < 0 and vmax < 0:
		plt_cmap = colormap_blue
	elif vmin < 0 and vmax >= 0:
		plt_cmap = colormap_divergent
		
	#Normalize values and create cmap
	norm_func = plt.Normalize(vmin=vmin, vmax=vmax)
	sm = plt.cm.ScalarMappable(cmap=plt_cmap, norm=norm_func)
	cmap = sm.get_cmap()
	color_func = lambda value: _rgb_to_hex(cmap(norm_func(value), bytes=True)[:3])
	return(color_func)
	

def network(network, 
				color_node_by=None,
				color_edge_by=None,
				size_node_by=None, 
				size_edge_by=None,
				engine="sfdp", 
				size="8,8", 
				save=None):
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
	save : str, optional
		Path to save network figure to. Format is inferred from the filename - if not valid, the default format is '.pdf'.	
	
	Raises
	-------
	TypeError
		If network is not a 
	ValueError 
		If any of 'color_node_by', 'color_edge_by' or 'size_edge_by' is not in node/edge attributes, or if 'engine' is not a valid graphviz engine.
	"""
		
	############### Test input ###############
	
	#Read nodes and edges from graph
	node_view = network.nodes(data=True)
	edge_view = network.edges(data=True)

	node_attributes = list(list(node_view)[0][-1].keys())    
	edge_attributes = list(list(edge_view)[0][-1].keys())
	
	if engine not in graphviz.ENGINES:
		raise ValueError("The given engine '{0}' is not in graphviz available engines: {1}".format(engine, graphviz.ENGINES))
	
	#todo: check size with re
	
	############### Initialize graph ###############
	
	dot = graphviz.Graph(engine=engine)
	dot.attr(size=size)
	
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
		na, nb = 14, 20
		map_value["node_size"] = lambda value: np.round((value-nmin)/(nmax-nmin)*(nb-na)+na, 2)
	
	#Edge size
	if size_edge_by != None:
		all_values = [edge[-1][size_edge_by] for edge in edge_view]
		vmin, vmax = np.min(all_values), np.max(all_values)
		a, b = 1, 8
		map_value["edge_size"] = lambda value: np.round((value-vmin)/(vmax-vmin)*(b-a)+a, 2)

	
	############### Add nodes to network ##############
	for node in node_view:
	
		node_name = node[0]
		node_att = node[1]
		
		attributes = {} #attributes for dot

		#Set node color
		if color_node_by != None:
			attributes["style"] = "filled"     
			value = node_att[color_node_by]
			attributes["color"] = map_value["node_color"](value)
			
		#Set node size
		if size_node_by != None:
			value = node_att[size_node_by]
			attributes["fontsize"] = str(map_value["node_size"](value))
		
		#After collecting all attributes; add node with attribute dict
		if len(attributes) > 0:
			dot.node(node_name, _attributes=attributes)
		else:
			dot.node(node_name)

	############### Add edges to network ###############
	for edge in edge_view:
	
		node1, node2 = edge[:2]
		edge_att = edge[-1]
		
		attributes = {}
		
		#Set edge color
		if color_edge_by != None:
			value = edge_att[color_edge_by]
			attributes["color"] = map_value["edge_color"](value)
		
		#Set edge size
		if size_edge_by != None:
			value = edge_att[size_edge_by]
			attributes["penwidth"] = str(map_value["edge_size"](value))
	   
		dot.edge(node1, node2, _attributes=attributes)
	
	############### Save to file ###############
	if save != None:
		splt = os.path.splitext(save)
		file_prefix = "".join(splt[:-1])
		fmt = splt[-1].replace(".", "")
		
		if fmt not in graphviz.FORMATS:
			fmt = "pdf"
		dot.render(filename=file_prefix, format=fmt, cleanup=True)
	
	return(dot)
