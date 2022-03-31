
import json
import time
from pathlib import Path


import yaml
import numpy as np
import time
from alive_progress import alive_bar
import json
import matplotlib.pyplot as plt
import numpy as np


from logging_files.settings import ExperimentSettings, ArchiveSettings, EmittersSettings, LoggingSettings
from ribs.archives import CVTArchive, GridArchive
from ribs.emitters import (GaussianEmitter, ImprovementEmitter, IsoLineEmitter,
                           OptimizingEmitter, RandomDirectionEmitter)
from ribs.optimizers import Optimizer
from ribs.visualize import cvt_archive_heatmap, grid_archive_heatmap



def sphere_evaluate(sol):
    """Sphere function evaluation and BCs for a batch of solutions.

    Args:
        sol (np.ndarray): (batch_size, dim) array of solutions
    Returns:
        objs (np.ndarray): (batch_size,) array of objective values
        bcs (np.ndarray): (batch_size, 2) array of behavior values
    """
    dim = sol.shape[1]

    # Shift the Sphere function so that the optimal value is at x_i = 2.048.
    sphere_shift = 5.12 * 0.4

    # Normalize the objective to the range [0, 100] where 100 is optimal.
    best_obj = 0.0
    worst_obj = (-5.12 - sphere_shift)**2 * dim
    raw_obj = np.sum(np.square(sol - sphere_shift), axis=1)
    objs = (raw_obj - worst_obj) / (best_obj - worst_obj) * 100

    # Calculate BCs.
    clipped = sol.copy()
    clip_indices = np.where(np.logical_or(clipped > 5.12, clipped < -5.12))
    clipped[clip_indices] = 5.12 / clipped[clip_indices]
    bcs = np.concatenate(
        (
            np.sum(clipped[:, :dim // 2], axis=1, keepdims=True),
            np.sum(clipped[:, dim // 2:], axis=1, keepdims=True),
        ),
        axis=1,
    )

    return objs, bcs

def save_heatmap(archive, heatmap_path):
    """Saves a heatmap of the archive to the given path.

    Args:
        archive (GridArchive or CVTArchive): The archive to save.
        heatmap_path: Image path for the heatmap.
    """
    if isinstance(archive, GridArchive):
        plt.figure(figsize=(8, 6))
        grid_archive_heatmap(archive, vmin=0, vmax=100)
        plt.tight_layout()
        plt.savefig(heatmap_path)
    elif isinstance(archive, CVTArchive):
        plt.figure(figsize=(16, 12))
        cvt_archive_heatmap(archive, vmin=0, vmax=100)
        plt.tight_layout()
        plt.savefig(heatmap_path)
    plt.close(plt.gcf())

def set_total_emitter_counts(gaussian, improvement, iso_line, optimizing, random_direction):
    return gaussian + improvement + iso_line + optimizing + random_direction

def make_emitter_seeds(seed, total_num_emitters):
        emitter_seeds = []
        if seed is None:
            emitter_seeds = [None] * total_num_emitters
        else:
            emitter_seeds = list(range(seed, seed + total_num_emitters))
        return emitter_seeds


def make_emitters(archive, emittersSettings, exp_settings, archive_settings):
    

    # Improvement:  
    # archive, initial_sol, 0.5,      batch_size=batch_size,    selection_rule=selection_rule,      seed=s
    # archive, x0,          sigma0,                             selection_rule="filter",            restart_rule="no_improvement", weight_rule="truncation", bounds=None, batch_size=None, seed=None
    # archive, initial_sol, 0.5,      batch_size=batch_size,    selection_rule=selection_rule,      seed=s
    population_size = exp_settings.population_size
    x0 = emittersSettings.initial_sol 
    sigma0 = emittersSettings.sigma0
    iso_sigma = emittersSettings.iso_sigma
    line_sigma = emittersSettings.line_sigma

    bounds = None

    gaussian_pop_size = emitter_settings.gaussian_pop_size
    isoline_pop_size = emitter_settings.isoline_pop_size
    imp_pop_size = emitter_settings.imp_pop_size
    rd_pop_size = emitter_settings.rd_pop_size
    opt_pop_size = emitter_settings.opt_pop_size

    imp_restart_rule = emitter_settings.imp_restart_rule
    imp_selection_rule = emitter_settings.imp_selection_rule
    imp_weight_rule = emitter_settings.imp_weight_rule

    rd_restart_rule = emitter_settings.rd_restart_rule
    rd_selection_rule = emitter_settings.rd_selection_rule
    rd_weight_rule = emitter_settings.rd_weight_rule

    opt_restart_rule = emitter_settings.opt_restart_rule
    opt_selection_rule = emitter_settings.opt_selection_rule
    opt_weight_rule = emitter_settings.opt_weight_rule
    


    emitters = []
    emitter_seeds = emittersSettings.emitter_seeds
    seed_index = 0

    for i in range(emittersSettings.gaussian):
        # archive, x0, sigma0, bounds=None, batch_size=64, seed=None
        emitters.append(GaussianEmitter(archive, x0, sigma0, bounds, gaussian_pop_size, emitter_seeds[seed_index]))
        seed_index += 1
    for i in range(emittersSettings.iso_line):
        # archive, x0, iso_sigma=0.01, line_sigma=0.2, bounds=None, batch_size=64, seed=None):
        emitters.append(IsoLineEmitter(archive, x0, iso_sigma, line_sigma, bounds, isoline_pop_size, emitter_seeds[seed_index]))        
        seed_index += 1
    for i in range(emittersSettings.improvement):
        emitters.append(ImprovementEmitter(archive, x0, sigma0, selection_rule=imp_selection_rule, restart_rule=imp_restart_rule, 
                                            weight_rule=imp_weight_rule, bounds = bounds, batch_size=imp_pop_size, seed = emitter_seeds[seed_index]))
        seed_index += 1
    for i in range(emittersSettings.random_direction):
        emitters.append(RandomDirectionEmitter(archive, x0, sigma0, rd_selection_rule, rd_restart_rule, rd_weight_rule,bounds, rd_pop_size, emitter_seeds[seed_index]))
        seed_index += 1
    for i in range(emittersSettings.optimizing):
        emitters.append(OptimizingEmitter(archive, x0, sigma0, opt_selection_rule, opt_restart_rule, opt_weight_rule, bounds, opt_pop_size, emitter_seeds[seed_index]))
        seed_index += 1
    return emitters
   
def make_archive(archiveSettings):
    if 'GridArchive' in archiveSettings.gridTypeClass:
        archive = GridArchive(archiveSettings.gridDimensions, archiveSettings.gridBounds, archiveSettings.seed)
    return archive

def make_optimizer(archive, emitters):
    return Optimizer(archive, emitters)

def load_yaml():
    stream = open("args.yaml", 'r')
    dictionary = yaml.safe_load(stream)


    experiment_settings = ExperimentSettings(dictionary['alg'], dictionary['population_size'], dictionary['generations'], dictionary.get('solution_dimension') )
    archive_settings = ArchiveSettings(dictionary['gridTypeClass'], dictionary['gridDimensions'], dictionary['maxBound'])
    emitter_settings = EmittersSettings(
        dictionary['gaussian'],dictionary['improvement'],dictionary['iso_line'],dictionary['optimizing'],
        dictionary['random_direction'], set_total_emitter_counts, make_emitter_seeds, 
        dictionary['solution_dimension'], dictionary['sigma0'],
        dictionary['gaussian_pop_size'], dictionary['isoline_pop_size'], dictionary['iso_sigma'], dictionary['line_sigma'],
        dictionary['rd_selection_rule'], dictionary['rd_restart_rule'], dictionary['rd_weight_rule'], 
        dictionary['rd_pop_size'], dictionary['imp_selection_rule'], dictionary['imp_restart_rule'], dictionary['imp_weight_rule'],
        dictionary['imp_pop_size'], dictionary['opt_selection_rule'], dictionary['opt_restart_rule'], dictionary['opt_weight_rule'],
        dictionary['opt_pop_size'])

        
    logging_settings = LoggingSettings(dictionary['outdir'], dictionary['log_freq'], dictionary['seed-log'], 0)
        
    return experiment_settings, archive_settings, emitter_settings, logging_settings

def logging_and_output(archive,generation,name,metrics,outdir,log_freq, final_gen):
    
    if generation % log_freq == 0 or final_gen:
        if final_gen:
            archive.as_pandas(include_solutions=final_gen).to_csv(outdir / f"{name}_archive.csv")

        # Record and display metrics.
        metrics["QD Score"]["x"].append(generation)
        metrics["QD Score"]["y"].append(archive.stats.qd_score)
        metrics["Archive Coverage"]["x"].append(generation)
        metrics["Archive Coverage"]["y"].append(archive.stats.coverage)
        print(f"Iteration {generation} | Archive Coverage: "
                f"{metrics['Archive Coverage']['y'][-1] * 100:.3f}% "
                f"QD Score: {metrics['QD Score']['y'][-1]:.3f}")

        save_heatmap(archive,
                        str(outdir / f"{name}_heatmap_{generation:05d}.png"))

def plot_metrics_end(name,metrics,outdir,non_logging_time):
    # Plot metrics.
    print(f"Algorithm Time (Excludes Logging and Setup): {non_logging_time}s")
    for metric in metrics:
        plt.plot(metrics[metric]["x"], metrics[metric]["y"])
        plt.title(metric)
        plt.xlabel("Iteration")
        plt.savefig(
            str(outdir / f"{name}_{metric.lower().replace(' ', '_')}.png"))
        plt.clf()
    with (outdir / f"{name}_metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)

if __name__ == '__main__':
   
    experiment_settings, archive_settings, emitter_settings, logging_settings = load_yaml()
    archive = make_archive(archive_settings)
    emitters = make_emitters(archive, emitter_settings, experiment_settings, archive_settings)
    solver = make_optimizer(archive,emitters)
    
   
    generations = experiment_settings.generations
    non_logging_time = logging_settings.non_logging_time

    with alive_bar(generations) as progress:
        #save_heatmap(archive, None, np.array([0,0]), str(outputDir / f"{experiment_settings.name}_heatmap_{0:05d}.png"))
        save_heatmap(archive, str(logging_settings.outdir / f"{experiment_settings.name}_heatmap_{0:05d}.png"))
        for gen in range(1, generations + 1):
            itr_start = time.time()
            sols = []
            sols = solver.ask()
            objs, bcs = sphere_evaluate(sols)
            solver.tell(objs, bcs)
            non_logging_time += time.time() - itr_start
            progress()

            final_gen = gen == generations

            if gen % logging_settings.log_freq == 0 or final_gen:
                logging_and_output(archive, gen, experiment_settings.name,experiment_settings.metrics, logging_settings.outdir, logging_settings.log_freq, final_gen)
                
           
    plot_metrics_end(experiment_settings.name,experiment_settings.metrics,logging_settings.outdir,non_logging_time)


