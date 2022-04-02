import json
import time
from pathlib import Path
import yaml
import numpy as np
from alive_progress import alive_bar
import matplotlib.pyplot as plt
import fire

from ribs.archives import CVTArchive, GridArchive
from ribs.emitters import (GaussianEmitter, ImprovementEmitter, IsoLineEmitter,
                           OptimizingEmitter, RandomDirectionEmitter)
from ribs.optimizers import Optimizer
from ribs.visualize import cvt_archive_heatmap, grid_archive_heatmap

from logging_files.settings import ExperimentSettings, ArchiveSettings, EmittersSettings, GaussianEmitterSettings, IsolineEmitterSettings, OptimizingEmitterSettings, ImprovementEmitterSettings, RandomDirectionEmitterSettings, LoggingSettings
from ribs.archives import CVTArchive, GridArchive
from ribs.emitters import (GaussianEmitter, ImprovementEmitter, IsoLineEmitter,
                           OptimizingEmitter, RandomDirectionEmitter)
from ribs.optimizers import Optimizer
from ribs.visualize import cvt_archive_heatmap, grid_archive_heatmap

from make_gif import main



def L1_evaluate(sol):
    """Sphere function evaluation and BCs for a batch of solutions.

    Args:
        sol (np.ndarray): (batch_size, dim) array of solutions
    Returns:
        objs (np.ndarray): (batch_size,) array of objective values
        bcs (np.ndarray): (batch_size, 2) array of behavior values
    """


    """
    L1: <x> in \mathbf{R} [-100, 100]
    f(x) = 50-x**2
    f(0) = 50, f(100), f(-100) = -9950
    
    max_bound = exp_settings['solution_dimension'] / 2 * 5.12
    """
    
    dim = 1


    # Normalize the objective to the range [0, 100] where 100 is optimal.
    best_obj = 50
    worst_obj = (50 - (100**2))*dim
    raw_obj = 50 - (sol**2)
    objs = (raw_obj - worst_obj) / (best_obj - worst_obj) * 100

    # Calculate BCs.
    clipped = sol.copy()
    fit = objs.copy()
    #clip_indices = np.where(np.logical_or(clipped > 100.0, clipped < -100.0))
    #clipped[clip_indices] = 100 / clipped[clip_indices]

    bcs = np.concatenate(
        (
            clipped,
            fit,

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

def set_total_emitter_counts(gaussianEmitterSettings, improvementEmitterSettings, iso_lineEmitterSettings, optimizingEmitterSettings, random_directionEmitterSettings):
    return gaussianEmitterSettings.count + improvementEmitterSettings.count + iso_lineEmitterSettings.count + optimizingEmitterSettings.count + random_directionEmitterSettings.count

def make_emitter_seeds(seed, total_num_emitters):
        emitter_seeds = []
        if seed is None:
            emitter_seeds = [None] * total_num_emitters
        else:
            emitter_seeds = list(range(seed, seed + total_num_emitters))
        return emitter_seeds


def make_emitters(archive, emittersSettings, exp_settings, archive_settings):


    x0 = emittersSettings.initial_sol 
    sigma0 = emittersSettings.sigma0
    gaussianSettings = emittersSettings.gaussianSettings
    isolineSettings = emittersSettings.isolineSettings
    optimizingSettings = emittersSettings.optimizingSettings
    improvementSettings = emittersSettings.improvementSettings
    randomDirectionSettings = emittersSettings.randomDirectionSettings
    bounds = None

    emitters = []
    emitter_seeds = emittersSettings.emitter_seeds
    seed_index = 0

    for i in range(gaussianSettings.count):
        emitters.append(GaussianEmitter(archive, x0, sigma0, bounds, gaussianSettings.pop_size, emitter_seeds[seed_index]))
        seed_index += 1
    for i in range(isolineSettings.count):
        emitters.append(IsoLineEmitter(archive, x0, isolineSettings.iso_sigma, isolineSettings.line_sigma, bounds, isolineSettings.pop_size, emitter_seeds[seed_index]))        
        seed_index += 1
    for i in range(improvementSettings.count):
        emitters.append(ImprovementEmitter(archive, x0, sigma0, selection_rule=improvementSettings.selection_rule, restart_rule=improvementSettings.restart_rule, 
                                            weight_rule=improvementSettings.weight_rule, bounds = bounds, batch_size=improvementSettings.pop_size, seed = emitter_seeds[seed_index]))
        seed_index += 1
    for i in range(randomDirectionSettings.count):
        emitters.append(RandomDirectionEmitter(archive, x0, sigma0, randomDirectionSettings.selection_rule, randomDirectionSettings.restart_rule, randomDirectionSettings.weight_rule,bounds, randomDirectionSettings.pop_size, emitter_seeds[seed_index]))
        seed_index += 1
    for i in range(optimizingSettings.count):
        emitters.append(OptimizingEmitter(archive, x0, sigma0, optimizingSettings.selection_rule, optimizingSettings.restart_rule, optimizingSettings.weight_rule, bounds, optimizingSettings.pop_size, emitter_seeds[seed_index]))
        seed_index += 1
    return emitters
   
def make_archive(archiveSettings):
    if 'GridArchive' in archiveSettings.gridTypeClass:
        archive = GridArchive(archiveSettings.gridDimensions, archiveSettings.gridBounds, archiveSettings.seed)
    return archive

def make_optimizer(archive, emitters):
    return Optimizer(archive, emitters)

def load_yaml():
    stream = open("L1_args.yaml", 'r')
    dictionary = yaml.safe_load(stream)


    experiment_settings = ExperimentSettings(dictionary['alg'], dictionary['generations'], dictionary.get('solution_dimension'))
    gridBounds = [tuple(v) for v in dictionary["gridBounds"]]

    archive_settings = ArchiveSettings(dictionary['gridTypeClass'], dictionary['gridDimensions'], gridBounds)

    optimizingEmitterSettings = OptimizingEmitterSettings(dictionary['optimizing'], dictionary['sigma0'],dictionary['opt_pop_size'],dictionary['opt_selection_rule'], dictionary['opt_restart_rule'], dictionary['opt_weight_rule'])
    improvementEmitterSettings = ImprovementEmitterSettings(dictionary['improvement'],dictionary['sigma0'],dictionary['imp_pop_size'],dictionary['imp_selection_rule'], dictionary['imp_restart_rule'], dictionary['imp_weight_rule'])
    randomDirectionEmitterSettings = RandomDirectionEmitterSettings(dictionary['random_direction'],dictionary['sigma0'],dictionary['rd_pop_size'],dictionary['rd_selection_rule'], dictionary['rd_restart_rule'], dictionary['rd_weight_rule'])
    gaussianEmitterSettings = GaussianEmitterSettings(dictionary['gaussian'], dictionary['sigma0'], dictionary['gaussian_pop_size'])
    isolineEmitterSettings = IsolineEmitterSettings(dictionary['iso_line'],dictionary['isoline_pop_size'], dictionary['iso_sigma'], dictionary['line_sigma'])
  

    emitter_settings = EmittersSettings(dictionary['solution_dimension'], dictionary['sigma0'], optimizingEmitterSettings, improvementEmitterSettings, randomDirectionEmitterSettings, gaussianEmitterSettings, isolineEmitterSettings, set_total_emitter_counts, make_emitter_seeds, None)
    logging_settings = LoggingSettings(dictionary['outdir'], dictionary['log_freq'], 0)
        
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
            objs, bcs = L1_evaluate(sols)
            #j = 0
            #for sol in sols:
                #print(f"Solution {j}: {sol[0]:.03f}, Obj:{objs[j][0]:02f}, BC:({bcs[j][0]:03f}, {bcs[j][1]:03f})")
            #    j = j + 1
            solver.tell(objs, bcs)
            non_logging_time += time.time() - itr_start
            progress()

            final_gen = gen == generations

            if gen % logging_settings.log_freq == 0 or final_gen:
                logging_and_output(archive, gen, experiment_settings.name,experiment_settings.metrics, logging_settings.outdir, logging_settings.log_freq, final_gen)
                
           
    plot_metrics_end(experiment_settings.name,experiment_settings.metrics,logging_settings.outdir,non_logging_time)

    import shutil
    from PIL import Image, ImageDraw, ImageFont
    import os
    import argparse
    import imageio
    shutil.copyfile("L1_args.yaml", f"{logging_settings.outdir}/L1_args.yaml")
    main(logging_settings.outdir)