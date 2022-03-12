
from datetime import datetime
import copy
import multiprocessing as mp
import pandas as pd
import random
import itertools
import scipy
import re

#Network analysis
import networkx as nx
import community as community_louvain

import tfcomb.utils
from tfcomb.logging import TFcombLogger, InputError
from tfcomb.utils import check_columns, check_type, check_string

#-------------------------------------------------------------------------------#
#------------------------ Build network from table -----------------------------#
#-------------------------------------------------------------------------------#

def _is_symmetric(a, rtol=1e-05, atol=1e-08):
	""" Utility to check if a matrix is symmetric. 
	Source: https://stackoverflow.com/a/42913743
	"""
	return np.allclose(a, a.T, rtol=rtol, atol=atol)


def _establish_node_attributes(table, node):
	""" 
	Returns a list of columns fitting to 'node' values.

	Parameters
	----------
	table : pd.DataFrame 
		Edges table including node/edge attributes.
	node : str
		Name of column contain node names.

	Returns
	-------
	list
		List of column names
	"""	

	node_attributes = []
	columns_to_assign = [col for col in table.columns if col != node]

	sub = table[:100000] #subset in interest of performance
	factorized = sub.apply(lambda x : pd.factorize(x)[0]) + 1 #factorize to enable correlation
	
	for attribute in columns_to_assign:
		try:
			p = scipy.stats.chisquare(factorized[node], f_exp=factorized[attribute])[1]
		except ValueError: #observed frequencies != expected frequencies
			p = 0.0 #not correlated
		except FloatingPointError: #sum of factorized is 0 -> all nan
			p = 0.0

		if p == 1.0: #columns are fully correlated; save to node attribute
			node_attributes.append(attribute)
	
	return(node_attributes)

def _get_table_dtypes(table):
	""" """
	columns = table.columns
	column_dtypes = table.dtypes.values
	dtype_list = [re.sub(r'[0-9]+', '', str(dtype)) for dtype in column_dtypes]
	dtype_list = [dtype if dtype != "object" else "string" for dtype in dtype_list]
	dtype_dict = dict(zip(columns, dtype_list))

	return(dtype_dict)

def build_network(edge_table, 
					node1="TF1", 
					node2="TF2", 
					node_table=None,
					directed=False, 
					multi=False,
					tool="networkx",
					verbosity=1):
	""" Build a network objecct from a table using either 'networkx' or 'graph-tool'.
	
	Parameters
	-----------
	edge_table : pd.DataFrame 
		Table containing rows of edges and edge information between node1/node2.
	node1 : str, optional
		The column to use as node1 ID. Default: "TF1".
	node2 : str, optional
		The column to use as node2 ID. Default: "TF2".
	node_table : pandas.DataFrame 
		A table of attributes to use for nodes. Default: node attributes are estimated from the columns in edge_table.
	directed : bool, optional
		Whether edges are directed or not. Default: False.
	multi : bool, optional
		Allow multiple edges between two vertices. NOTE: Only valid for tool == 'networkx'. If False, the first occurrence of TF1-TF2/TF2-TF1 in the table is used. Default: False.
	tool : str, optional
		Which module to use for generating network. Must be one of 'networkx' or 'graph-tool'. Default: 'networkx'.
	verbosity : int, optional
		Verbosity of logging (0/1/2/3). Default: 1.

	Returns
	--------
	if tool is 'networkx': networkx.Graph / networkx.DiGraph / networkx.MultiGraph / networkx.MultiDiGraph - depending on parameters given.
	if tool is 'graph-tool': graph_tool.Graph 
	"""

	#TODO: check given input
	check_type(edge_table, pd.DataFrame, "table")
	check_string(tool, ["networkx", "graph-tool"], "tool")

	if tool == "graph-tool":
		if tfcomb.utils.check_graphtool() == True: #check if graph-tool is installed
			import graph_tool

	#Setup table
	table = edge_table.copy()
	check_columns(table, [node1, node2])

	#Setup logger
	logger = TFcombLogger(verbosity)

	#########################################
	############# Prepare edges #############
	#########################################

	if multi == True and tool == "graph-tool":
		raise InputError("The option 'multi=True' is not compatible with 'tool=graph-tool'. Please adjust parameters.")

	# Subset edges if multi is not allowed
	if multi == False:

		#Remove duplicates of the same edge (first occurrence is kept)
		table = table.drop_duplicates([node1, node2])

		#Collect unique pairs (first occurrence is kept)
		table.set_index([node1, node2], inplace=True)
		pairs = table.index
		to_keep = {}
		for pair in pairs:
			if not pair[::-1] in to_keep: #if opposite was not already found
				to_keep[pair] = ""

		#Subset table
		table = table.loc[list(to_keep.keys())]
		table.reset_index(inplace=True)

		logger.spam("Subset edges (head): {0}".format(table.head()))

	#########################################
	####### Setup network attributes ########
	#########################################

	attribute_columns = [col for col in table.columns if col not in [node1, node2]]

	if node_table is None:
		
		#Establish node attributes
		node1_attributes = _establish_node_attributes(table, node1)
		logger.debug("node1_attributes: {0}".format(node1_attributes))
		
		node2_attributes = _establish_node_attributes(table, node2)
		node2_attributes = list(set(node2_attributes) - set(node1_attributes)) #prevent the same columns from being assigned to both TF1 and TF2)
		logger.debug("node2_attributes: {0}".format(node2_attributes))

		#Setup tables for node1 and node2 information
		node1_table = table[[node1] + node1_attributes].set_index(node1, drop=False) #also includes node1
		node2_table = table[[node2] + node2_attributes].set_index(node2, drop=False) #also includes node2

		#Merge node information to dict for network
		node_table = node1_table.merge(node2_table, left_index=True, right_index=True, how="outer")
		node_table.drop_duplicates(inplace=True)
	
	else:
		#TODO: check that node_table fits with node1/node2
		pass

	logger.spam("node_table head:\n{0}".format(node_table.head()))
	node_attributes = list(node_table.columns)
	logger.debug("node_attributes: {0}".format(node_attributes))
	node_attribute_dict = {i: {att: row[att] for att in node_attributes} for i, row in node_table.iterrows()}
	logger.spam("node_attribute_dict: {0} (...)".format({i: node_attribute_dict[i] for i in list(node_attribute_dict.keys())[:5]}))
	
	######## Setup edge attributes #######	
	edge_attributes = [col for col in attribute_columns if col not in node_attribute_dict]
	logger.debug("edge_attributes: {0}".format(edge_attributes))
	edges_list = [(row[node1], row[node2], {att: row[att] for att in edge_attributes}) for i, row in table.iterrows()]
	logger.spam("edges_list: {0} (...)".format(edges_list[:3]))

	#########################################
	############## Build Graph ##############
	#########################################

	if tool == "networkx":
		if multi == True:
			if directed == True:
				G = nx.MultiDiGraph()
			else:
				G = nx.MultiGraph()
		else:
			if directed == True:
				G = nx.DiGraph()
			else:
				G = nx.Graph()

		#Add collected edges
		G.add_edges_from(edges_list)
		
		#Add node attributes
		nx.set_node_attributes(G, node_attribute_dict)


	elif tool == "graph-tool":

		#Initialize graph
		g = graph_tool.all.Graph(directed=directed)

		#Node attributes
		node_attributes = [node1, node2] + node_attributes
		#TODO: remove duplicate node_attributes when node1/node2 were already part of attributes
		dtype_dict = _get_table_dtypes(node_table)
		for att in node_attributes:
			eprop = g.new_vertex_property(dtype_dict[att])
			g.vertex_properties[att] = eprop

		#Edge attributes
		dtype_dict = _get_table_dtypes(table[edge_attributes])
		for att in edge_attributes:
			eprop = g.new_edge_property(dtype_dict[att])
			g.edge_properties[att] = eprop

		## Add nodes with properties
		name2idx = {} #TF name to idx
		#idx2name = {}
		for i, row in node_table.to_dict(orient="index").items():
			v = g.add_vertex()
			
			name = i #index is the name of node (TF)
			name2idx[name] = v #idx of node
			#idx2name[int(v)] = name
			
			for prop in g.vertex_properties:
				g.vertex_properties[prop][v] = row[prop]

		## Add edges with properties
		for i, row in table.to_dict(orient="index").items(): #loop over all edges in table
			v1, v2 = name2idx[row[node1]], name2idx[row[node2]]
			e = g.add_edge(v1, v2)
			
			for prop in g.edge_properties:
				g.edge_properties[prop][e] = row[prop]

	#Return finished network
	if tool == "networkx":
		return(G)
	elif tool == "graph-tool":
		return(g)


#-------------------------------------------------------------------------------#
#------------------------- Network analysis algorithms -------------------------#
#-------------------------------------------------------------------------------#

def get_degree(G, weight=None, direction="both"):
	"""
	Get degree per node in graph. If weight is given, the degree is the sum of weighted edges.

	Parameters
	-----------
	G : networkx.Graph
		An instance of networkx.Graph
	weight : str, optional
		Name of an edge attribute within network. Default: None.
	direction : str, optional
		Which edge direction to use for calculating degrees. Can be one of: ["both", "in", "out"]. Default: 'both'.

	Returns
	--------
	DataFrame
		A table of format (...)

	"""

	#Check input
	tfcomb.utils.check_type(G, [nx.Graph])
	tfcomb.utils.check_type(weight, [str, type(None)], "weight")
	tfcomb.utils.check_string(direction, ["both", "in", "out"], direction)

	
	if weight is None:

		if direction == "both":
			unweighted = dict(G.degree())
		elif direction == "in":
			unweighted = dict(G.in_degree())
		elif direction == "out":
			unweighted = dict(G.out_degree())
		
		df = pd.DataFrame.from_dict(unweighted, orient="index")
		
	else:
		tfcomb.utils.check_type(weight, [str], "weight")

		edge_attributes = list(list(G.edges(data=True))[0][-1].keys())
		if weight in edge_attributes:

			if direction == "both":
				weighted = dict(G.degree(weight=weight))
			elif direction == "in":
				weighted = dict(G.in_degree(weight=weight))
			elif direction == "out":
				weighted = dict(G.out_degree(weight=weight))

			df = pd.DataFrame.from_dict(weighted, orient="index")
			
		else:
			raise ValueError("Weight '{0}' is not an edge attribute of given network. Available attributes are: {1}".format(weight, edge_attributes))
	
	df.columns = ["degree"] 
	df.sort_values("degree", inplace=True, ascending=False)    

	return(df)

#Graph partitioning 
def partition_louvain(G, weight=None, logger=None):
	"""
	Partition a network using community louvain. Sets the attribute

	Parameters
	----------
	G : networkx.Graph 
		An instance of a network graph to partition
	weight : str
		Attribute in graph to use as weight. The higher the weight, the stronger the link. Default: None.
	attribute_name : str
		The attribute name to use for saving partition. Default: "partition".
	logger : 

	Returns
	--------

	"""

	if logger is None:
		logger = TFcombLogger(0)

	#TODO: check 
	tfcomb.utils.check_type(G, [nx.Graph])

	#network must be undirected
	if G.is_directed():
		raise TypeError("Bad graph type, use only non directed graph")

	#Process weights
	edge_view = G.edges(data=True)
	edge_attributes = list(list(edge_view)[0][-1].keys())
	if weight is None:
		#choose a weight name which is not within network to ensure that all weights are set to 1. 
		#Ref: https://github.com/taynaud/python-louvain/issues/73#issuecomment-751483227

		weight = "None"
		while weight in edge_attributes: #if weight was in edge_attributes, get random string
			weight = tfcomb.utils.random_string()
		

	else:
		
		edge_view = G.edges(data=True)
		edge_attributes = list(list(edge_view)[0][-1].keys())

		#Check whether weight is available as edge attribute
		if weight not in edge_attributes:
			raise ValueError("Weight '{0}' is not an edge attribute in network. Available edge attributes are: {1}".format(weight, edge_attributes))
		
		#convert all weights into 0-1

	logger.debug("'weight' is set to: '{0}'".format(weight))	
		
	#Partition network
	logger.debug("Running community_louvain.best_partition()")
	partition_dict = community_louvain.best_partition(G, weight=weight, random_state=1) #random_state ensures that results are reproducible
	partition_dict_fmt = {key: {"partition": str(value + 1)} for key, value in partition_dict.items()}

	#Add partition information to each node
	for node_i in partition_dict_fmt:
		G.nodes[node_i]["partition"] = partition_dict_fmt[node_i]["partition"]
	#nx.set_node_attributes(G, partition_dict_fmt) #overwrites previous attributes; solved by loop over dict

	#No return - G is changed in place

def partition_blockmodel(g):
	""" Partitioning of a graph tool graph using stochastic block model minimization.
	
	Parameters
	-----------
	g : a graph.tool graph
	
	"""
	
	#check if graph-tool is installed
	if tfcomb.utils.check_graphtool() == True:
		import graph_tool

	#Infer blocks
	state = graph_tool.inference.minimize.minimize_blockmodel_dl(g)
	blocks = state.get_blocks()

	#Add vertex property to graph
	partition_prop = g.new_vertex_property("string")
	g.vertex_properties["partition"] = partition_prop
	
	n_nodes = g.num_vertices()
	for i in range(n_nodes):
		g.vertex_properties["partition"][i] = blocks[i]

	#No return - g was changed in place


def get_node_table(G):
	""" Get a table containing node names and node attributes for G.

	Parameters
	-----------
	G : a networkx Graph object or graph_tool Graph object
	
	Returns
	--------
	pandas.DataFrame
	"""

	if isinstance(G, nx.Graph):
		nodeview = G.nodes(data=True)
		table  = pd.DataFrame().from_dict(dict(nodeview), orient='index')
	else:
		#check if graph-tool is installed
		if tfcomb.utils.check_graphtool() == True:
			import graph_tool

		if isinstance(G, graph_tool.Graph):
	
			#Information about graph
			n_nodes = G.num_vertices()
			properties = list(G.vertex_properties.keys())
		

			data = {}
			for i in range(n_nodes):
				data[i] = {prop: G.vertex_properties[prop][i] for prop in properties}

			table = pd.DataFrame.from_dict(data, orient="index")
		
		else:
			raise ValueError("Unknown format of input Graph")

	return(table)

def get_edge_table(G):
	""" Get a table containing edge names and edge attributes for G.

	Parameters
	-----------
	G : a networkx Graph object 
	
	Returns
	--------
	pandas.DataFrame
	"""

	edgeview = G.edges(data=True)
	d = {(e[0],e[1]):e[2] for e in list(edgeview)} #dict of (TF1,TF2):{edge_att}
	table  = pd.DataFrame().from_dict(d, orient='index')

	return(table)

def create_random_network(nodes, edges):
	""" 
	Create a random network with the given nodes and the number of edges.

	nodes : list
		List of nodes to use in network.
	edges : int
		Number of edges between nodes.
	"""

	G_rand = nx.Graph()
	G_rand.add_nodes_from(nodes)
	
	#Setup edges
	combis = list(itertools.combinations(nodes, 2))
	edges_list = random.choices(combis, k=edges)
	G_rand.add_edges_from(edges_list)
	
	return(G_rand)

def powerlaw():
	""" """