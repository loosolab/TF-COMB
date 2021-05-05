import nxviz
import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns


def bubble(edges_table, yaxis="confidence", color_by="lift", size_by="TF1_TF2_support", figsize=(6,6), ax=None):
	""" 
	Plot bubble plot with TF1-TF2 pairs on the x-axis and a choice of measure on the y-axis, as well as color and size of bubbles. 

	Parameters
	----------
	edges_table : pandas.DataFrame
		Dataframe containing 
	yaxis : str, optional
		Defaults to "confidence"
	color_by : str
		Defaults to "lift"
	size_by : str
		Defaults to "TF1_TF2_support"
	figsize : tuple


	Return
	-------


	"""

	#Setup figure
	if ax == None:
		fig, ax = plt.subplots(figsize=figsize) 

	with sns.axes_style("whitegrid"):
		
		ax = sns.scatterplot(
							data=edges_table,
							ax=ax,
							x=edges_table.index, 
							y=yaxis, 
							hue=color_by, 
							size=size_by,
							palette="PuBu", 
							edgecolor=".7",
							#height=6, 
							#sizes=(50, 250),
							#aspect=1.5
		)

	ax.grid()

	labels = list(edges_table.index)

	# Tweak the figure to finalize
	ax.set(ylabel=yaxis, xlabel="Co-occurring pairs")
	ax.set_xticklabels(labels, rotation=45, ha="right")

	return(ax)


def heatmap(rules_table, columns="TF1", rows="TF2", color_by="cosine", figsize=(6,6)):
	"""
	Plot heatmap with TF1 and TF2 on rows and columns respectively. Heatmap colormap is chosen by .color_by.

	Parameters
	----------
	rules_table : pandas.DataFrame
	columns : str, optional
	rows : str, optional
	color_by : str, optional
	"""

	#Test input format
	

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
	#fig, ax = plt.subplots(figsize=figsize) 
	h = sns.clustermap(pivot_table, 
								mask=mask,
								cbar=True, 
								cmap=cmap,
								center=center,
								cbar_kws={'label': color_by}, 
								xticklabels=True,
								yticklabels=True
					)

	xticklabels = h.ax_heatmap.axes.get_xticklabels()
	yticklabels = h.ax_heatmap.axes.get_yticklabels()

	h.ax_heatmap.axes.set_xticklabels(xticklabels, rotation=45, ha="right")
	h.ax_heatmap.axes.set_yticklabels(yticklabels, rotation=0)
	h.ax_heatmap.axes.set_facecolor('lightgrey') #color of NA-values

	#plt.title("Top {0} association rules".format(n_rules))
	#plt.tight_layout()

	#return(fig)

def volcano(table, measure=None, pvalue=None, measure_threshold=None, pvalue_threshold=None):
	"""
	Plot volcano-style plots combining one measure and the pvalue

	Parameters
	-----------
	table : pd.DataFrame
		
	measure : str
		The measure to show on the x-axis
	pvalue : str
		The column containing pvalues

	measure_threshold : float or list of floats

	pvalue_threshold : float between 0-1
		Default is 0.05
	"""

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

	

def plot_go_enrichment(table, aspect="MF", n_terms=20, threshold=0.05):
	"""
	
	Parameters
	--------------
	table : pandas.DataFrame 
		The output of tfcomb.analysis.go_enrichment 
	aspect : str
		One of ["MF", "BP", ""]


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


def network(G, layout="spring_layout", color_edge_by=None, color_node_by=None, figsize=(6,6)):
	""" 
	Plot network of a networkx object.

	Parameters
	-----------
	network : networkx.Graph

	layout : str
		Must be one of ....
	color_edge_by : str
		

	color_node_by : str

	Raises
	-------
	TypeError
		if network is not a 
	KeyError
		#
	ValueError 
		If layout string is not valid

	See also
	--------
	tfcomb.analysis.build_network()

	"""

	################# Check input validity #################

	available_layouts = [
		#"bipartite_layout",
		"circular_layout",
		"kamada_kawai_layout",
		#"random_layout",
		#"rescale_layout",
		#"rescale_layout_dict",
		"shell_layout",
		"spring_layout",
		"spectral_layout",
		#"planar_layout",
		"fruchterman_reingold_layout",
		"spiral_layout",
		#"multipartite_layout",
	]
	
	if layout not in available_layouts:
		raise ValueError("Layout '{0}' is not a valid networkx layout".format(layout))

	#Check if input network is a graph

	#available edge attributes
	edge_attributes = list(list(G.edges(data=True))[0][-1].keys())
	node_attributes = list(list(G.nodes(data=True))[0][-1].keys())
	
	for edge_attribute in [color_edge_by]:
		if edge_attribute is not None:
			if edge_attribute not in edge_attributes:
				raise ValueError()
	for node_attribute in [color_node_by]:
		if node_attribute is not None:
			if node_attribute not in node_attributes:
				raise ValueError()

	
	################# Decide what colormaps to use based on data ##################

	colormap_red = _truncate_colormap(plt.cm.Reds, 0.3)
	colormap_blue = _truncate_colormap(plt.cm.Blues_r, maxval=0.7)

	#Establish color of edge
	edge_cmap = None
	edge_vmin, edge_vmax = None, None
	if color_edge_by is None:
		edge_color = "red"
	else:
		edge_weights = [data[color_edge_by] for (_, _, data) in G.edges.data()]
		node_vmin = np.min(edge_weights)
		node_vmax = np.max(edge_weights)

		if np.min(edge_weights) < 0 and np.max(edge_weights) > 0: #divergent colormap
			edge_cmap = plt.cm.bwr
			
			#Ensure center at 0
			abs_max = np.max(np.abs(edge_weights))
			edge_vmin = -abs_max
			edge_vmax = abs_max
		elif np.min(edge_weights) < 0:
			edge_cmap = colormap_blue
		else:
			edge_cmap = colormap_red

		edge_color = edge_weights

	#Establish color of nodes
	node_cmap = None
	node_vmin, node_vmax = None, None 
	if color_node_by == None:
		node_color = "grey"
	else:
		node_weights = [data[color_node_by] for (_, data) in G.nodes.data()]
		node_vmin = np.min(node_weights)
		node_vmax = np.max(node_weights)
		
		if np.min(node_weights) < 0 and np.max(node_weights) > 0: #divergent colormap
			node_cmap = plt.cm.coolwarm
			
			#Ensure center at 0
			abs_max = np.max(np.abs(node_weights))
			node_vmin = -abs_max
			node_vmax = abs_max
		elif np.min(node_weights) < 0:
			node_cmap = colormap_blue
		else:
			node_cmap = colormap_red

		node_color = node_weights


	######################## Draw network graph ########################

	#Setup figure
	fig, ax = plt.subplots(figsize=figsize, frameon=False)

	#Establish layout
	layout_function = getattr(nx, layout)
	pos = layout_function(G, scale=1.5)

	#Draw nodes
	nx_nodes = nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=node_cmap, #node_size=node_size,
													 vmin=node_vmin, vmax=node_vmax) #, node_size=0, alpha=0.4, edge_color="r", font_size=16, with_labels=True)
	
	#Draw node labels
	y_off = 0.15
	nx.draw_networkx_labels(G, pos = {k:([v[0], v[1]+y_off]) for k,v in pos.items()})
	
	#Draw edges
	nx_edges = nx.draw_networkx_edges(G, pos, edge_color = edge_color, edge_cmap = edge_cmap) # node_size=node_size)
	

	####################### Add colorbars to plot ##########################

	if color_edge_by != None:
		sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=edge_vmin, vmax=edge_vmax))
		sm.set_array([])
		cbar = plt.colorbar(sm)
		cbar.ax.set_title("Edge color:\n" + color_edge_by, rotation=45, ha="left")
		
	if color_node_by != None:
		sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=node_vmin, vmax=node_vmax))
		sm.set_array([])
		cbar = plt.colorbar(sm)
		cbar.ax.set_title("Node color:\n" + color_node_by, rotation=45, ha="left")

	ax.set_aspect(1)
	ax.axis('off')
	#plt.tight_layout()
	#return(fig)


def circos(network, color_edge_by="lift", size_edge_by="lift", color_node_by=None, size_node_by=None):
	"""

	Plot circos plot from networkx network object

	Parameters
	-----------
	color_edge_by : str
		string must be the name of an edge attribute
	size_edge_by : str
		string must be the name of an edge attribute
	color_node_by : str
		string must be the name of a node attribute
	size_node_by : str
		string must be the name of a node attribute
	
	Returns
	--------

	"""

	#G = tfcomb.analysis.build_network(edges_table) #, node1=node1, node2=node2)
	#edge_width = edges_table[size_edge_by]

	#Plot circos of G
	c = nxviz.CircosPlot(network, node_labels=True, fontfamily="sans-serif", edge_width=size_edge_by) #, edge_width=size_edge_by)  #, node_color='affiliation', node_grouping='affiliation')
	c.draw()	
