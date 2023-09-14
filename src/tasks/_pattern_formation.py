from jax._src.config import NoDefault
from src.models._model import State

from typing import Callable, Optional, NamedTuple
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn

import equinox as eqx
from jaxtyping import Float, PyTree, Array

def gaussian(x, mu, a, sigma):
    return a * jnp.exp(-(jnp.square(x-mu).sum() / (2*sigma)**2))

def functional_form(pts: jax.Array, a: jax.Array, sigma: float=.1):
    """Return the functional form of a set of points"""
    def f(x: jax.Array):
         return jax.vmap(gaussian, in_axes=(None, 0, 0, None))(x, pts, a, sigma).sum(0)
    return f


class FFLoss:
	#-------------------------------------------------------------------
	def __init__(self, target, eval_pts=None, sigma=.1):
		if eval_pts is None:
			self.eval_pts = target
		else: 
			self.eval_pts = jnp.concatenate([eval_pts, target], axis=0)
		self.target_ff = functional_form(target, jnp.ones((target.shape[0],)), sigma)
		self.sigma = sigma
	#-------------------------------------------------------------------
	def __call__(self, pts: Float[Array, "N 2"], mask: Float[Array, "N"])->float:
		pts_ff = functional_form(pts, mask, sigma=self.sigma)
		eval_pts = jnp.concatenate([self.eval_pts, pts], axis=0)
		yt = jax.vmap(self.target_ff)(eval_pts)
		y = jax.vmap(pts_ff)(eval_pts)

		return jnp.square(y-yt).mean()
	#-------------------------------------------------------------------


class PatternFormationTask(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	loss_fn: Callable[[Float[Array, "N 2"], Float[Array, "N"]], float]
	statics: PyTree
	devo_steps: int
	N_particles: int
	particle_hidden_dims: int
	#-------------------------------------------------------------------

	def __init__(self, statics: PyTree, target: Float[Array, "Nt 2"], sigma: float=.1,
				 eval_pts: Optional[Float[Array, "Ne 2"]]=None, devo_steps: int=20,
				 N_particles: int=100, particle_hidden_dims: int=8):
		
		self.statics = statics
		self.loss_fn = FFLoss(target, eval_pts, sigma)
		self.devo_steps = devo_steps
		self.N_particles = N_particles
		self.particle_hidden_dims = particle_hidden_dims

	#-------------------------------------------------------------------

	def __call__(self, params: PyTree, key: jr.PRNGKeyArray)->float:
		
		key_init, key_roll = jr.split(key)
		model = eqx.combine(params, self.statics)
		state, _ = model.rollout(self.init_state(key_init), key_roll, 
									  self.devo_steps)
		pattern_loss = self.loss_fn(state.p, state.mask)

		return pattern_loss

	#-------------------------------------------------------------------

	def init_state(self, key: jr.PRNGKeyArray)->State:
		
		kp = key
		return State(
			p = jr.normal(kp, (self.N_particles, 2)),
			h = jnp.zeros((self.N_particles, self.particle_hidden_dims)),
			rec=jnp.array([0]*(self.N_particles*5)),
			send=jnp.array([1]*(self.N_particles*5)),
			divs=jnp.zeros((self.N_particles,)),
			mask=jnp.ones((self.N_particles,))
		)















