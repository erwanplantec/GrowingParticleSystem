from typing import NamedTuple
from src.models._model import State

import jax
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx
from jaxtyping import Float, PyTree, Array

class Goal(NamedTuple):
	goal: Float[Array, "Dg"]
	goal_mask: Float[Array, "N"]

class DyadicTask(eqx.Module):
	
	""" 
	Goal is a tuple (target_positions, number of divs)
	Only particle 1 can see the goal
	There is an edge from 1 to 0 to communicate
	fitness is the mse between goal and final state
	"""
	#-------------------------------------------------------------------
	particle_hidden_dims: int
	statics: PyTree
	rollout_steps: int
	goal_type: str
	#-------------------------------------------------------------------

	def __init__(self, statics: PyTree[...], particle_hidden_dims: int,
				 rollout_steps: int=20, goal_type: str = "xn"):
		
		self.statics = statics
		self.particle_hidden_dims = particle_hidden_dims
		self.rollout_steps = rollout_steps
		self.goal_type = goal_type

	#-------------------------------------------------------------------

	def __call__(self, params: PyTree, key: jr.PRNGKeyArray)->float:
		
		model = eqx.combine(params, self.statics)

		key_state, key_rollout = jr.split(key)
		init_state = self.init_state(key_state)
		end_state, states = model.rollout(init_state, key_rollout, self.rollout_steps)

		g = {}
		g["x"] = end_state.p[0] #(2,)
		g["n"] = states.divs.sum(0)[0, None] #(1,)
		g = jnp.concatenate([g[_g] for _g in self.goal_type], axis=-1)
		l = jnp.square(init_state.aux.goal - g).mean() 

		return l


	#-------------------------------------------------------------------

	def sample_goal(self, key: jr.PRNGKeyArray)->jax.Array:

		key_xy, key_n = jr.split(key)
		g = {}
		g["x"] = jr.normal(key_xy, (2,))
		g["n"] = jr.randint(key_n, (1,), minval=0, maxval=10).astype(float)
		g = jnp.concatenate([g[_g] for _g in self.goal_type], axis=-1)
		return g

	#-------------------------------------------------------------------

	def init_state(self, key)->State:
		
		kp, kg = jr.split(key, 2)
		return State(
			p = jr.normal(kp, (2, 2)),
			h = jnp.zeros((2, self.particle_hidden_dims)),
			rec=jnp.array([0]),
			send=jnp.array([1]),
			divs=jnp.zeros((2,)),
			aux = Goal(goal=self.sample_goal(kg),goal_mask=jnp.array([0., 1.]),)
		)