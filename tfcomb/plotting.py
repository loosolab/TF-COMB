import nxviz
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def bubble(edges_table, yaxis="confidence", color_by="lift", size_by="TF1_TF2_support", figsize=(7,7), ax=None):
	
	#setup figure
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


def heatmap(edges_table, columns="TF1", rows="TF2", color_by="lift", figsize=(8,8)):
	"""
	"""

	# Create support table for the heatmap
	heatmap_table = edges_table.pivot(index=rows, columns=columns, values=color_by)

	#Mask any NA values
	mask = np.zeros_like(heatmap_table)
	mask[np.isnan(heatmap_table)] = True

	#Heatmap
	fig, ax = plt.subplots(figsize=figsize) 
	h = sns.heatmap(heatmap_table, ax=ax, annot=False, 
													mask=mask,
													cbar=True, 
													cmap="PuBu",
													cbar_kws={'label': color_by}, 
													xticklabels=True,
													yticklabels=True
					)

	h.set_xticklabels(h.get_xticklabels(), rotation=45, ha="right")
	h.set_yticklabels(h.get_yticklabels(), rotation=0)
	h.set_facecolor('lightgrey') #color of NA-values

	#plt.title("Top {0} association rules".format(n_rules))
	plt.tight_layout()

	return(fig)


def plot_go_enrichment(table, aspect="MF", n_terms=20, threshold=0.05):
	"""
	
		
	Parameters:
	----------
	table (Pandas DataFrame): The output of tfcomb.analysis.go_enrichment 
		
	Returns:
	----------


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


def _plot_condition_heatmap(edges_table):
	
	pass


#setup network
def _build_network(table, node1="TF1", node2="TF2", weight=None):
	"""

	"""

	#Check that node1, node2 and weight are present
	chosen_columns = [col for col in [node1, node2, weight] if col != None]
	available_columns = table.columns
	for col in chosen_columns:
		if col not in available_columns:
			raise ValueError("{0} is not a column in the given table. Available columns are: {1}".format(col, available_columns))

	node1_list = set(table[node1])
	node2_list = set(table[node2])

	#Select edge with highest weight

	#Depending on directional
	edge_attributes = set(table.columns) - set([node1, node2])
	#print(edge_attributes)

	#Setup edges list
	edges = [(row[node1], row[node2], {att: row[att] for att in edge_attributes}) for i, row in table.iterrows()]

	#Set weight
	if weight != None:
		for i in range(len(edges)):
			edges[i][2]["weight"] = edges[i][2][weight] #create key with "weight"

	#Add node attributes

	#Create directed multigraph
	MG = nx.MultiDiGraph()
	MG.add_edges_from(edges)

	#Create undirected graph
	#self.nx_undirected = []

	return(MG)


def circos(edges_table, node1="TF1", node2="TF2",
						color_edge_by="lift", size_edge_by="lift", color_node_by=None, size_node_by=None):
	"""

	"""
	
	G = _build_network(edges_table, node1=node1, node2=node2)


	#edge_width = edges_table[size_edge_by]

	#Plot circos
	c = nxviz.CircosPlot(G, node_labels=True, fontfamily="sans-serif", edge_width=size_edge_by) #, edge_width=size_edge_by)  #, node_color='affiliation', node_grouping='affiliation')
	c.draw()	
