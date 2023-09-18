from jaxtyping import PyTree
from src.models._model import ParticleSystem, State
from src.tasks._dyadic import DyadicTask
from src.training._evo import EvosaxTrainer
from src.training._utils import pickle_save

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn

import equinox as eqx
import evosax as ex

import matplotlib.pyplot as plt

GOAL_TYPE = "xn"

GOAL_DIMS = 3
HIDDEN_DIMS = 8
MSG_DIMS = 16
ROLL_STEPS = 20

POPSIZE = 256
GENS = 500
REPS = 20

SAVE_FILE = "saves/test.pickle"

def init_trainer(task, params):
	shaper = ex.ParameterReshaper(params)
	es = ex.DES(popsize=POPSIZE, num_dims=shaper.total_params)
	trainer = EvosaxTrainer(task, es, shaper, gens=GENS, n_repeats=REPS)
	return trainer

def init_task(statics: PyTree):
	task = DyadicTask(statics, HIDDEN_DIMS, ROLL_STEPS, goal_type=GOAL_TYPE, homeostasis=True)
	return task

def init_model(key):
	aux_getter = lambda s: s.aux.goal[None, :] * s.aux.goal_mask[:, None]
	model = ParticleSystem(hidden_dims=HIDDEN_DIMS, msg_dims=MSG_DIMS, aux_dims=GOAL_DIMS, aux_getter=aux_getter, key=key)
	return model

def show_results(data):
	fits = data["fitnesses"].min(-1)
	bests_fit = data["states"].best_fitness
	plt.scatter(jnp.arange(fits.shape[0]), fits, alpha=.3, s=10)
	plt.plot(bests_fit, color="r")
	plt.show()
def main():
	key = jr.PRNGKey(10101)

	model = init_model(key)
	params, statics = eqx.partition(model, eqx.is_array)

	task = init_task(statics)

	trainer = init_trainer(task, params)

	best_params, data = trainer(key)

	show_results(data)

	pickle_save(SAVE_FILE, {"best_params":best_params, "data": data})

if __name__ == '__main__':
	main()