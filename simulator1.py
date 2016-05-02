import graph
import networkx as nx
import numpy as np
import small_world
import pylab as plt
from matplotlib import pyplot as pp
import scipy

# The simulator class
class Simulator(object):
	def __init__(self, initial_prefs, network_structure='small_world', num_nodes=100, num_nbhds=2, num_iters=100, graph_args={}, agent_args={'k': 0., 'n': 1., 'heterogeneous': False, 'stubborn_prop': 0.}, budget=1):
		self.network_structure = network_structure
		self.num_nodes = num_nodes
		if budget != 2:
			self.num_nbhds = num_nbhds
		else:
			self.num_nbhds = num_nbhds + 1
		self.num_iters = num_iters
		self.initial_prefs = initial_prefs
		self.graph_args = graph_args
		self.agent_args = agent_args
		self.agents = self.initialize_agents()
		self.nx_graph, self.graph = self.gen_graph()
		self.nx_graph_pos = nx.spring_layout(self.nx_graph, scale=2) 
		self.budget=budget

	def initialize_agents(self):
		agents = []
		for i in range(self.num_nodes):
			stubborn = False
			if np.random.rand() <= self.agent_args['stubborn_prop']:
				stubborn = True
			if self.agent_args['heterogeneous']:
				k = abs(np.random.normal(loc=0.0, scale=2.0))
				n = abs(np.random.normal(loc=1.0, scale=1.0))
				agent = graph.Agent(node_id=i, n_nbhds=self.num_nbhds, nbhd=np.random.choice(self.num_nbhds, p=self.initial_prefs), k=k, n=n, stubborn=stubborn)
			else:
				agent = graph.Agent(node_id=i, n_nbhds=self.num_nbhds, nbhd=np.random.choice(self.num_nbhds, p=self.initial_prefs), k=self.agent_args['k'], n=self.agent_args['n'], stubborn=stubborn)
			agents.append(agent)
		return agents 

	def convert_nx_graph(self, nx_graph):
		graph = {}
		for node in nx_graph:
			graph[node] = nx_graph.neighbors(node)
		return graph

	# Should return an adjacency list
	def gen_graph(self):
		if self.network_structure == 'small_world':
			nx_graph = nx.watts_strogatz_graph(self.num_nodes, self.graph_args['k'], self.graph_args['p'])
			return nx_graph, self.convert_nx_graph(nx_graph)
		elif self.network_structure == 'erdos_renyi':
			nx_graph = nx.fast_gnp_random_graph(self.num_nodes, self.graph_args['p'])
			return nx_graph, self.convert_nx_graph(nx_graph)
		elif self.network_structure == 'preferential_attachment':
			nx_graph = nx.barabasi_albert_graph(self.num_nodes, self.graph_args['m'])
			return nx_graph, self.convert_nx_graph(nx_graph)

	def simulate(self, plot_stats=True):
		# Re-initialize at the beginning of each simulation
		self.agents = self.initialize_agents()
		self.nx_graph, self.graph = self.gen_graph()
		self.nx_graph_pos = nx.spring_layout(self.nx_graph) 

		simulation_graph = graph.Network(self.agents, self.graph, self.budget)
		# print [agent.id for agent in self.agents]
		# print self.graph.keys()
		nbhd_proportions = []
		homeless_proportion = []
		price_0 = []
		price_1 = []
		for i in range(self.num_iters):
			self.plot(i)
			# print len([agent.nbhd for agent in self.agents if agent.nbhd == 0]), len([agent.nbhd for agent in self.agents if agent.nbhd == 1])
			
			nbhd_proportions.append(self.nbhd_proportions())
			homeless_proportion.append(self.homeless_pop())
			if self.budget:
				price_0.append(simulation_graph.neighborhood_prices[0])
				price_1.append(simulation_graph.neighborhood_prices[1])
			simulation_graph.update()
			self.agents = simulation_graph.agents
		if (plot_stats):
			# Plot neighborhood proportion 
			pp.plot([i for i in range(self.num_iters)], [nbhd[0] for nbhd in nbhd_proportions], 'bo', label='Neighborhood 0')			
			pp.plot([i for i in range(self.num_iters)], [nbhd[1] for nbhd in nbhd_proportions], 'ro', label='Neighborhood 1')
			pp.ylabel('Proportion of nodes in each neighborhood')
			pp.xlabel('Time')
			pp.legend()
			pp.show()
		else:
			return nbhd_proportions, homeless_proportion, price_0, price_1, self.avg_deg_nbhd_changes(), self.end_result()

	# Runs n simulations and plots average summary statistics
	def simulate_n(self, n):
		print 'new simulation'
		nbhd_proportions_diff = []
		homeless_proportion = []
		average_price_0 = []
		average_price_1 = []
		end_results = []
		master_dict = {} # to keep track of average degree/neighborhood changes

		for i in range(n):
			nbhd_proportions_i, homeless_proportion_i, price_0_i, price_1_i, deg_nbhd_dict, end_result = self.simulate(plot_stats=False)
			nbhd_proportions_diff.append([abs(nbhd[0] - nbhd[1]) for nbhd in nbhd_proportions_i])
			homeless_proportion.append(homeless_proportion_i)
			for deg in deg_nbhd_dict:
				if deg in master_dict:
					master_dict[deg].append(deg_nbhd_dict[deg])
				else:
					master_dict[deg] = [deg_nbhd_dict[deg]]

			if self.budget:
				average_price_0.append(price_0_i)
				average_price_1.append(price_1_i)
			end_results.append(end_result)

		nbhd_proportions_np = np.array(nbhd_proportions_diff)
		homeless_proportion_np = np.array(homeless_proportion)

		for deg in master_dict:
			master_dict[deg] = float(sum(master_dict[deg]))/len(master_dict[deg])

		if self.budget:
			average_price_0_np = np.array(average_price_0)
			average_price_1_np = np.array(average_price_1)


		# Begin plotting and saving
		# Variables for naming figures
		proportions = 'evensplit'
		if self.initial_prefs[0] == .1:
			proportions = '.1_.9_split'
		elif self.initial_prefs[0] == .25:
			proportions = '.25_.75_split'
		elif self.initial_prefs[0] == .4:
			proportions = '.4_.6_split'

		parameters = ''
		if self.network_structure == 'small_world':
			parameters = 'k=%.2f_p=%.2f' % (self.graph_args['k'], self.graph_args['p'])
		elif self.network_structure == 'erdos_renyi':
			parameters = 'p=%.2f' % self.graph_args['p']
		elif self.network_structure == 'preferential_attachment':
			parameters = 'm=%d' % self.graph_args['m']
		pp.clf()

		# Absolute Proportion Difference
		pp.plot([i for i in range(self.num_iters)], nbhd_proportions_np.mean(0), 'bo')
		pp.ylabel('Absolute difference in proportions between neighborhoods')
		pp.xlabel('Time')
		# pp.show()
		filename = ''
		if self.agent_args['heterogeneous']:
			filename = 'Plots/Heterogeneous Agents/%s/%diters_%dtrials_%s_proportions_%s.png' % (self.network_structure, self.num_iters, n, proportions, parameters)
		elif self.budget == 1:
			filename = 'Plots/With Budgets/%s/%diters_%dtrials_%s_proportions_%s.png' % (self.network_structure, self.num_iters, n, proportions, parameters)
		elif self.budget == 2:
			filename = 'Plots/Renting/%s/%diters_%dtrials_%s_proportions_%s.png' % (self.network_structure, self.num_iters, n, proportions, parameters)
		elif self.agent_args['stubborn_prop'] > 0:
			filename = 'Plots/Stubborn/%s/%diters_%dtrials_%s_proportions_%s_stubborn=%f.png' % (self.network_structure, self.num_iters, n, proportions, parameters, self.agent_args['stubborn_prop'])
		else:
			filename = 'Plots/No Budgets/%s/%diters_%dtrials_%s_proportions_%s.png' % (self.network_structure, self.num_iters, n, proportions, parameters)
		pp.savefig(filename, format="PNG")
		pp.clf()

		# Ultimate converge results:
		yvalues = [0, 0, 0]
		for result in end_results:
			yvalues[result + 1] += 1
		x = scipy.arange(3)
		y = scipy.array(yvalues)
		f = plt.figure()
		ax = f.add_axes([.1, .1, .8, .8])
		ax.bar(x, y, align='center')
		ax.set_xticks(x)
		ax.set_xticklabels(['No convergence', 'All 0', 'All 1'])
		filename = ''
		if self.agent_args['heterogeneous']:
			filename = 'Plots/Heterogeneous Agents/%s/%diters_%dtrials_%s_finalresult_%s.png' % (self.network_structure, self.num_iters, n, proportions, parameters)
		elif self.budget == 1:
			filename = 'Plots/With Budgets/%s/%diters_%dtrials_%s_finalresult_%s.png' % (self.network_structure, self.num_iters, n, proportions, parameters)
		elif self.budget == 2:
			filename = 'Plots/Renting/%s/%diters_%dtrials_%s_finalresult_%s.png' % (self.network_structure, self.num_iters, n, proportions, parameters)
		elif self.agent_args['stubborn_prop'] > 0:
			filename = 'Plots/Stubborn/%s/%diters_%dtrials_%s_finalresult_%s_stubborn=%f.png' % (self.network_structure, self.num_iters, n, proportions, parameters, self.agent_args['stubborn_prop'])
		else:
			filename = 'Plots/No Budgets/%s/%diters_%dtrials_%s_finalresult_%s.png' % (self.network_structure, self.num_iters, n, proportions, parameters)
		plt.savefig(filename, format="PNG")
		plt.clf()

		# Average degree/neighborhood switching
		pp.plot([i for i in master_dict], [master_dict[i] for i in master_dict], 'bo')
		pp.ylabel('Average number of neighborhood swaps')
		pp.xlabel('Node degree')
		filename = ''
		if self.agent_args['heterogeneous']:
			filename = 'Plots/Heterogeneous Agents/%s/%diters_%dtrials_%s_swaps_%s.png' % (self.network_structure, self.num_iters, n, proportions, parameters)
		elif self.budget == 1:
			filename = 'Plots/With Budgets/%s/%diters_%dtrials_%s_swaps_%s.png' % (self.network_structure, self.num_iters, n, proportions, parameters)
		elif self.budget == 2:
			filename = 'Plots/Renting/%s/%diters_%dtrials_%s_swaps_%s.png' % (self.network_structure, self.num_iters, n, proportions, parameters)
		elif self.agent_args['stubborn_prop'] > 0:
			filename = 'Plots/Stubborn/%s/%diters_%dtrials_%s_swaps_%s_stubborn=%f.png' % (self.network_structure, self.num_iters, n, proportions, parameters, self.agent_args['stubborn_prop'])
		else:
			filename = 'Plots/No Budgets/%s/%diters_%dtrials_%s_swaps_%s.png' % (self.network_structure, self.num_iters, n, proportions, parameters)
		pp.savefig(filename, format="PNG")
		pp.clf()

		# Average Neighborhood Price (only if budget is set)
		if self.budget == 1:
			pp.plot([i for i in range(self.num_iters)], average_price_0_np.mean(0), 'bo', label='Neighborhood 0')
			pp.plot([i for i in range(self.num_iters)], average_price_1_np.mean(0), 'ro', label='Neighborhood 1')
			pp.ylabel('Average neighborhood price')
			pp.xlabel('Time')
			pp.legend()
			filename = 'Plots/With Budgets/%s/%diters_%dtrials_%s_price_%s.png' % (self.network_structure, self.num_iters, n, proportions, parameters)
			pp.savefig(filename, format="PNG")
			pp.clf()


		# Average Neighborhood Price (renting)
		if self.budget == 2:
			pp.plot([i for i in range(self.num_iters)], average_price_0_np.mean(0), 'bo', label='Neighborhood 0')
			pp.plot([i for i in range(self.num_iters)], average_price_1_np.mean(0), 'ro', label='Neighborhood 1')
			pp.ylabel('Average neighborhood price')
			pp.xlabel('Time')
			pp.legend()
			filename = 'Plots/Renting/%s/%diters_%dtrials_%s_price_%s.png' % (self.network_structure, self.num_iters, n, proportions, parameters)
			pp.savefig(filename, format="PNG")
			pp.clf()
			pp.plot([i for i in range(self.num_iters)], homeless_proportion_np.mean(0), 'bo', label='Homelessness')
			pp.ylabel('Proportion homeless')
			pp.xlabel('Time')
			filename = 'Plots/Renting/%s/%diters_%dtrials_%s_homeless_%s.png' % (self.network_structure, self.num_iters, n, proportions, parameters)
			pp.savefig(filename, format="PNG")
			pp.clf()


	# Plots the current state of the graph with colors representing neighborhood choices
	# Right now works for 2 neighborhoods, could be extended to more
	def plot(self, i):
		nbhd_1 = [node for node in self.nx_graph if self.agents[node].nbhd == 1]
		nbhd_2 = [node for node in self.nx_graph if self.agents[node].nbhd == 0]
		nbhd_3 = [node for node in self.nx_graph if self.agents[node].nbhd == 2]
		labels = {}
		if self.budget:
			for node in self.nx_graph:
				labels[node] = '%.1f' % self.agents[node].agent_budget
		nx.draw_networkx_nodes(self.nx_graph, self.nx_graph_pos, nodelist=nbhd_1, node_color='r', node_size=200, alpha=.8)
		nx.draw_networkx_nodes(self.nx_graph, self.nx_graph_pos, nodelist=nbhd_2, node_color='g', node_size=200, alpha=.8)
		nx.draw_networkx_nodes(self.nx_graph, self.nx_graph_pos, nodelist=nbhd_3, node_color='b', node_size=200, alpha=.8)
		nx.draw_networkx_edges(self.nx_graph, self.nx_graph_pos, width=1.0, alpha=0.5)
		nx.draw_networkx_labels(self.nx_graph, self.nx_graph_pos, labels, font_size=8)

		filename = '%d'  % i
		pp.savefig(filename, format="PNG")
		pp.clf()

	# Returns the current distribution of neighborhood choices among agents
	def nbhd_proportions(self):
		counts = []
		for nbhd in range(self.num_nbhds):
			counts.append(sum([1 for n in self.agents if n.nbhd == nbhd]))
		return [float(p) / self.num_nodes for p in counts]

	# Returns the proportion of nodes that are homeless
	def homeless_pop(self):
		return float(sum([1 for n in self.agents if n.nbhd == 2]))/self.num_nodes

	# Returns the average degree of nodes who have chosen a particular neighborhood
	# Returns -1 if no nodes have chosen that neighborhood
	def average_degree(self, nbhd):
		degrees = [self.nx_graph.degree(n) for n in self.nx_graph if self.agents[n].nbhd == nbhd]
		if not degrees:
			return 0.0
		return float(sum(degrees)) / len(degrees)

	# Returns the average number of times a node changed its neighborhood choice
	def average_nbhd_changes(self):
		return sum([n.nbhd_changes for n in self.agents]) / float(self.num_nodes)

	def avg_deg_nbhd_changes(self):
		changes = {}
		for n in self.nx_graph:
			if self.nx_graph.degree(n) not in changes:
				changes[self.nx_graph.degree(n)] = [self.agents[n].nbhd_changes]
			else:
				changes[self.nx_graph.degree(n)].append(self.agents[n].nbhd_changes)
		for deg in changes:
			changes[deg] = float(sum(changes[deg]))/len(changes[deg])
		return changes

	def end_result(self):
		all_0 = True
		all_1 = True
		for n in self.agents:
			if n.nbhd == 1:
				all_0 = False
			else:
				all_1 = False
		if all_0:
			return 0
		elif all_1:
			return 1
		else:
			return -1

	# Returns a list mapping node to its degree and the number of times its changed neighborhoods
	def degree_and_num_changes(self):
		l = []
		for node in self.nx_graph:
			l.append((self.nx_graph.degree(node), self.agents[node].nbhd_changes))
		return l


initial_proportions = [[0.5, 0.5], [.25, .75], [.4, .6], [.1, .9]]
stubborn_proportions = [.2, .4, .5, .75, .9] 

network = 'small_world'
arg = {'k' : 5, 'p' : .1}

for proportions in initial_proportions:
	# stubborn
	for stubborn in stubborn_proportions:
		agent_arg = {'k':0., 'n':1., 'heterogeneous': False, 'stubborn_prop': stubborn}
		test_sim = Simulator(proportions, network_structure=network, graph_args=arg, agent_args=agent_arg, num_iters=500, budget=0)
		test_sim.simulate_n(100)
	# with budgets
	agent_arg = {'k':0., 'n':1., 'heterogeneous': False, 'stubborn_prop': 0.0}
	test_sim = Simulator(proportions, network_structure=network, graph_args=arg, agent_args=agent_arg, num_iters=500, budget=1)
	test_sim.simulate_n(100)

network = 'erdos_renyi'
arg = {'p': .2}
for proportions in initial_proportions:
	# stubborn
	for stubborn in stubborn_proportions:
		agent_arg = {'k':0., 'n':1., 'heterogeneous': False, 'stubborn_prop': stubborn}
		test_sim = Simulator(proportions, network_structure=network, graph_args=arg, agent_args=agent_arg, num_iters=100, budget=0)
		test_sim.simulate_n(100)
	# with budgets
	agent_arg = {'k':0., 'n':1., 'heterogeneous': False, 'stubborn_prop': 0.0}
	test_sim = Simulator(proportions, network_structure=network, graph_args=arg, agent_args=agent_arg, num_iters=100, budget=1)
	test_sim.simulate_n(100)

network = 'preferential_attachment'
arg = {'m': 5}
for proportions in initial_proportions:
	# stubborn
	for stubborn in stubborn_proportions:
		agent_arg = {'k':0., 'n':1., 'heterogeneous': False, 'stubborn_prop': stubborn}
		test_sim = Simulator(proportions, network_structure=network, graph_args=arg, agent_args=agent_arg, num_iters=200, budget=0)
		test_sim.simulate_n(100)
	# with budgets
	agent_arg = {'k':0., 'n':1., 'heterogeneous': False, 'stubborn_prop': 0.0}
	test_sim = Simulator(proportions, network_structure=network, graph_args=arg, agent_args=agent_arg, num_iters=200, budget=1)
	test_sim.simulate_n(100)
'''

proportions = [0.5, 0.5]
arg = {'k' : 5, 'p' : .1}
agent_arg = {'k':0., 'n':1., 'heterogeneous': True, 'stubborn_prop': 0.0}
test_sim = Simulator(proportions, network_structure='small_world', graph_args=arg, agent_args=agent_arg, num_nodes=10, num_iters=50, budget=0)
test_sim.simulate()
'''
