from ctypes import Union
from evosax.strategies import de
from src.models._base import BaseModel

import jax
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx 
import equinox.nn as nn

from typing import Callable, Collection, NamedTuple, Optional, Tuple, Union
from jaxtyping import Float, Array, Int, PyTree


class State(NamedTuple):
    p: Float[Array, "N Dp"]
    h: Float[Array, "N Dh"]
    rec: Int[Array, "E"]
    send: Int[Array, "E"] 
    divs: Float[Array, "N"]
    aux: Collection = {}
    mask: Optional[Float[Array, "N"]]=None

@jax.jit
def incr(x):
    n = x.sum().astype(int)
    return x.at[n].set(1.)

@jax.jit
def nincr(x, d):
    return jax.lax.fori_loop(0, d.sum().astype(int), lambda i, x: incr(x), x)


class ParticleSystem(BaseModel):
    
    """
    """
    #-------------------------------------------------------------------
    cell: nn.GRUCell
    pi: Union[nn.MLP, Callable]
    msg: Union[nn.MLP, Callable]
    connector: Union[Callable, PyTree[...]]
    has_aux: bool
    aux_getter: Optional[Callable[[State], Float[Array, "N ..."]]]
    spatial_encoder: Callable[[Float[Array, "2"]], Float[Array, "..."]]
    #-------------------------------------------------------------------

    def __init__(
        self, 
        hidden_dims: int, 
        msg_dims: int,
        *,
        key: jr.PRNGKeyArray,
        spatial_dims: int = 2,
        aux_dims: int=3,
        aux_getter: Optional[Callable[[State], Float[Array, "N ..."]]]=None,
        connector: Union[Callable, PyTree]=lambda s, *_: s,
        spatial_encoder: Callable=lambda x:x,
        spatial_encoding_dims: int=2,
        pi_fn: Optional[Callable]=None,
        msg_fn: Optional[Callable]=None):

        key_cell, key_pi, key_msg = jr.split(key, 3)

        self.cell = nn.GRUCell(msg_dims+aux_dims+spatial_encoding_dims, hidden_dims, key=key_cell)
        if pi_fn is None:
            self.pi = nn.MLP(hidden_dims, spatial_dims, 32, 1, key=key_pi)
        else: 
            self.pi  = pi_fn
        if msg_fn is None:
            self.msg = nn.MLP(hidden_dims+spatial_dims, msg_dims, 32, 1, key=key_msg)
        else:
            self.msg = msg_fn
        self.connector = connector
        self.has_aux = aux_dims > 0
        if self.has_aux: assert aux_getter is not None
        self.aux_getter = aux_getter
        self.spatial_encoder = spatial_encoder

    #-------------------------------------------------------------------

    def __call__(self, state: State, key: jr.PRNGKeyArray)->State:
        
        state = self.connector(state, key)
        h_send = state.h[state.send]
        d = state.p[state.send] - state.p[state.rec]
        m = jax.vmap(self.msg)(jnp.concatenate([h_send, d], axis=-1))
        m = jax.ops.segment_sum(m, state.rec, state.p.shape[0])
        pe = jax.vmap(self.spatial_encoder)(state.p)
        if self.has_aux:   
            assert self.aux_getter is not None
            aux = self.aux_getter(state)
            x = jnp.concatenate([m, pe, aux], axis=-1)
        else :
            x = jnp.concatenate([m, pe], axis=-1)
        h = jax.vmap(self.cell)(x, state.h)
        v = jax.vmap(self.pi)(h)
        d = (h[:, 0]>0.5).astype(float)
        h = jnp.where(d[:, None], h.at[:, 0].set(-1.), h)
        if state.mask is not None:
            h = h * state.mask[:,None]
            v = v * state.mask[:,None]

        return state._replace(h=h, p=state.p+v, divs=d)


class KNNConnector(eqx.Module):
    
    """
    Create edges to each node from its k nearest neighbors
    """
    #-------------------------------------------------------------------
    k: int
    #-------------------------------------------------------------------

    def __init__(self, k: int=5):
        
        self.k = k

    #-------------------------------------------------------------------

    def __call__(self, state: State, key: jr.PRNGKeyArray)->State:

        assert state.mask is not None
        
        max_nodes = state.p.shape[0]
        dp = state.p[:, None, :] - state.p
        d = (dp*dp).sum(-1)
        d = jnp.where(state.mask[None, :], d, jnp.inf)
        _, idxs = jax.lax.top_k(-d, self.k)

        s = jnp.where(state.mask[:, None], idxs, max_nodes-1)
        r = jnp.where(state.mask[:, None], jnp.mgrid[:max_nodes, :self.k][0], max_nodes-1)

        s = s.reshape((-1,))
        r = r.reshape((-1,))

        return state._replace(send=s, rec=r)


class GrowingParticleSystem(ParticleSystem):
    
    """
    """
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------

    def __call__(self, state: State, key: jr.PRNGKeyArray)->State:

        assert state.mask is not None
        
        ku, kp = jr.split(key)
        state = super().__call__(state, ku)
        state = self._add_nodes(state, kp)

        return state

    #-------------------------------------------------------------------

    def _add_nodes(self, state: State, key: jr.PRNGKeyArray)->State:

        assert state.mask is not None
        
        d = state.divs
        max_nodes = state.h.shape[0]
        nmask = nincr(state.mask, d)

        tgt = jnp.cumsum(d) * d - d
        tgt = jnp.where(d, tgt.astype(int), -1) + state.mask.sum().astype(int) * d.astype(int)
        mask_new = nmask * (1. - state.mask)

        np = jax.ops.segment_sum(state.p, tgt, max_nodes)
        np = np + (jr.normal(key, state.p.shape) * .001)
        np = jnp.where(mask_new[:, None], np, state.p)

        return state._replace(p=np, mask=nmask)



