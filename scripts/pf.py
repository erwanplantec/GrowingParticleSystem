from src.models._model import ParticleSystem, State, KNNConnector
from src.tasks._pattern_formation import PatternFormationTask
from src.training._evo import EvosaxTrainer
from src.tasks._utils import string_to_points
from src.training._utils import pickle_save, save_pytree

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn

import equinox as eqx
import evosax as ex

import matplotlib.pyplot as plt

TARGET = "U"

HIDDEN_DIMS = 8
MSG_DIMS = 16
ROLL_STEPS = 20
N = 50

POPSIZE = 64
GENS = 10
REPS = 1

SAVE_FILE = "saves/pf.eqx"

def init_model(key):
	connector = KNNConnector(k=5)
	model = ParticleSystem(hidden_dims=HIDDEN_DIMS, msg_dims=MSG_DIMS, aux_dims=0, connector=connector, key=key)
	return model

def init_task(statics)->PatternFormationTask:
	target = jnp.array(string_to_points(TARGET))
	return PatternFormationTask(statics, target, devo_steps=ROLL_STEPS, N_particles=N)

def init_trainer(task, params):
	shaper = ex.ParameterReshaper(params)
	es = ex.DES(popsize=POPSIZE, num_dims=shaper.total_params)
	trainer = EvosaxTrainer(task, es, shaper, gens=GENS, n_repeats=REPS)
	return trainer

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

	save_pytree(SAVE_FILE, {"best_params":eqx.combine(best_params, statics), "data": data})


if __name__ == '__main__':
	main()