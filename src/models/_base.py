import jax
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx 
import equinox.nn as nn

from typing import NamedTuple, Tuple
from jaxtyping import Float, Array, Int

class State(NamedTuple):
	pass

class BaseModel(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	#-------------------------------------------------------------------

	def __call__(self, state: State, key: jr.PRNGKeyArray)->State:
		
		raise NotImplementedError("call method has not been implemented")

	#-------------------------------------------------------------------

	def rollout(self, init_state: State, key: jr.PRNGKeyArray, n: int)->Tuple[State, State]:

		def _step(c, x):
			s, k = c 
			k, sk = jr.split(k)
			s = self.__call__(s, sk)
			return [s, k], s

		[s, _], ss = jax.lax.scan(_step, [init_state, key], None, n)
		return s, ss