import copy
import numpy as np

# The agent class
class Agent(object):
	def __init__(self, node_id, n_nbhds=2, k=0.0, n=1.0, nbhd=0, stubborn=False):
		self.id = node_id # The agent's id
		self.k = k # The constant value in the ant stigmergy model
		self.n = n # The power in the ant stigmergy model
		self.nbhd = nbhd # The agent's current neighborhood
		self.n_nbhds = n_nbhds # Number of neighborhoods
		self.nbhd_changes = 0
		self.agent_budget = max(0, np.random.normal(loc=1.0))
		self.stubborn = stubborn

	def normalize_vec(self, vec):
		denom = sum(vec)
		return [float(i)/denom for i in vec]

	def update_neighborhood(self, neighbors, prices, budget):
		nbhd_count = [0. for i in range(self.n_nbhds)]
		for neighbor in neighbors:
			nbhd_count[neighbor.nbhd] += 1
		n_neighbors = float(sum(nbhd_count))

		# If you have no neighbors, keep same neighborhood
		if n_neighbors == 0:
			return

		for n in range(2):
			#nbhd_count[n] = ((self.k + nbhd_count[n]) ** self.n )/((self.k + n_neighbors) ** self.n)
			nbhd_count[n] = ((self.k + nbhd_count[n]) ** self.n )
		nbhd_count = self.normalize_vec(nbhd_count)

		old_nbhd = self.nbhd
		self.nbhd = np.random.choice(len(nbhd_count), p = nbhd_count)
		if self.stubborn:
			stubborn_nbhd_count = [i/3.0 if i != old_nbhd else i for i in nbhd_count]
			stubborn_nbhd_count = self.normalize_vec(stubborn_nbhd_count)
			self.nbhd = np.random.choice(len(stubborn_nbhd_count), p = stubborn_nbhd_count)
		
		if budget == 2: # renting
			'''if self.agent_budget >= 3.0:
				print self.agent_budget, old_nbhd, self.nbhd
				print [(neighbor.nbhd, neighbor.agent_budget) for neighbor in neighbors]'''

			if self.nbhd == 2:
				if prices[0] <= self.agent_budget or prices[1] <= self.agent_budget:
					if prices[0] < prices[1]:
						self.nbhd = 0
					else:
						self.nbhd = 1

			elif prices[self.nbhd] > self.agent_budget: # can't afford to move
				if prices[old_nbhd] > self.agent_budget: #can't afford to stay
					if self.nbhd == old_nbhd:
						other_neighborhood = (self.nbhd + 1) % 2
						if prices[other_neighborhood] > self.agent_budget:
							self.nbhd = 2 # Homeless
						else:
							self.nbhd = other_neighborhood
					else:
						self.nbhd = 2
				else: 
					self.nbhd = old_nbhd

		if old_nbhd != self.nbhd:
			if budget == 1: #0 = none, 1 = buying, 2 = renting
				if self.agent_budget + prices[old_nbhd] < prices[self.nbhd]:
					# If change is not within budget
					self.agent_budget = self.agent_budget + prices[old_nbhd] - prices[self.nbhd]
					self.nbhd = old_nbhd
					return
			# If change is within budget, record that a change was made		
			self.nbhd_changes += 1

# The network class
class Network(object):
	def __init__(self, agents, graph, budget):
		self.agents = agents # Agents
		self.graph = graph # Graph structures
		if budget != 2:
			self.num_neighborhoods = 2
		else:
			self.num_neighborhoods = 3
		self.budget = budget
		self.neighborhood_pop = self.get_nbhd_pop()
		self.neighborhood_prices = {}
		self.update_prices() # Dict form

	def id_to_agent(self, i, fixed_agents):
		return fixed_agents[i]
	
	def get_nbhd_pop(self):
		population_counter = {}
		for nbhd in range(self.num_neighborhoods):
			population_counter[nbhd] = sum([1 for n in self.agents if n.nbhd == nbhd])
		return population_counter

	def update_prices(self):
		if self.budget == 1:
			self.neighborhood_prices[0] = 10. * self.neighborhood_pop[0]/(self.neighborhood_pop[0] + self.neighborhood_pop[1])
			self.neighborhood_prices[1] = 10. * self.neighborhood_pop[1]/(self.neighborhood_pop[0] + self.neighborhood_pop[1])
			self.neighborhood_prices[2] = 0.
		elif self.budget == 2:
			self.neighborhood_prices[0] = 2. * self.neighborhood_pop[0]/(self.neighborhood_pop[0] + self.neighborhood_pop[1])
			self.neighborhood_prices[1] = 2. * self.neighborhood_pop[1]/(self.neighborhood_pop[0] + self.neighborhood_pop[1])
			self.neighborhood_prices[2] = 0.

	def update(self):
		old_agents = copy.deepcopy(self.agents)
		new_agents = []
		for node_id in self.graph:
			agent = self.id_to_agent(node_id, old_agents)
			agent.update_neighborhood([self.id_to_agent(i, old_agents) for i in self.graph[node_id]], self.neighborhood_prices, self.budget)
			new_agents.append(agent)
		self.agents = new_agents
		self.neighborhood_pop = self.get_nbhd_pop()
		if self.budget != 0:
			self.update_prices()
			
'''agent0 = Agent(0, nbhd=0)
agent1 = Agent(1, nbhd=1)
agent2 = Agent(2, nbhd=0)
agent3 = Agent(3, nbhd=1)

agent0.update_neighborhood([agent1, agent2, agent3])
print 'neighborhood selection:'
print agent0.nbhd'''
