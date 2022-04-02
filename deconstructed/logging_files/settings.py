
from pathlib import Path
import numpy as np


class ExperimentSettings():
    def __init__(self, alg, generations, solution_dimension):
        self.alg = alg
        self.generations = generations
        self.solution_dimension = solution_dimension    
        self.metrics = {
                "QD Score": {"x": [0],"y": [0.0],},
                "Archive Coverage": {"x": [0],"y": [0.0],},}
        self.name = f"{str(self.alg)}_{str(self.solution_dimension)}"

class ArchiveSettings():
    def __init__(self,  gridTypeClass, gridDimensions , gridBounds, seed=None): 
        self.gridTypeClass = gridTypeClass
        self.gridDimensions = gridDimensions
        self.gridBounds = [gridBounds[0], gridBounds[1]]
        self.seed = seed

class LoggingSettings():
    def __init__(self, outdir,log_freq, non_logging_time):
        self.outdir = outdir
        self.log_freq = log_freq
        self.non_logging_time = non_logging_time

        self.outdir = Path(self.outdir)
        if not self.outdir.is_dir():
            self.outdir.mkdir()  
        

class GaussianEmitterSettings():
    def __init__(self, count, sigma0, gauss_pop_size):
        self.count = count
        self.sigma0 = sigma0
        self.pop_size = gauss_pop_size

class IsolineEmitterSettings():
    def __init__(self, count,  isoline_pop_size, iso_sigma, line_sigma):
        self.count = count
        self.pop_size = isoline_pop_size
        self.iso_sigma = iso_sigma
        self.line_sigma = line_sigma

class ImprovementEmitterSettings():
    def __init__(self, count, sigma0, imp_pop_size, imp_selection_rule, imp_restart_rule, imp_weight_rule):
        self.count = count
        self.sigma0 = sigma0
        self.pop_size = imp_pop_size
        self.selection_rule = imp_selection_rule
        self.restart_rule = imp_restart_rule
        self.weight_rule = imp_weight_rule

class RandomDirectionEmitterSettings():
    def __init__(self, count, sigma0, rd_pop_size, rd_selection_rule, rd_restart_rule, rd_weight_rule):
        self.count = count
        self.sigma0 = sigma0
        self.pop_size = rd_pop_size
        self.selection_rule = rd_selection_rule
        self.restart_rule = rd_restart_rule
        self.weight_rule = rd_weight_rule

class OptimizingEmitterSettings():
    def __init__(self, count, sigma0, opt_pop_size, opt_selection_rule, opt_restart_rule, opt_weight_rule):
        self.count = count
        self.sigma0 = sigma0
        self.pop_size = opt_pop_size
        self.selection_rule = opt_selection_rule
        self.restart_rule = opt_restart_rule
        self.weight_rule = opt_weight_rule


class EmittersSettings():
    def __init__(
        self, solution_dimension, sigma0, optimizingEmitterSettings, improvementEmitterSettings, 
        randomDirectionEmitterSettings, gaussianEmitterSettings, isolineEmitterSettings, 
        set_total_emitter_counts, make_emitter_seeds, seed = None):
        
        self.total_num_emitters = set_total_emitter_counts(gaussianEmitterSettings, improvementEmitterSettings, isolineEmitterSettings, optimizingEmitterSettings, randomDirectionEmitterSettings)
        self.seed = seed
        self.emitter_seeds = make_emitter_seeds(seed, self.total_num_emitters)
        
        self.solution_dimension = solution_dimension
        self.sigma0 = sigma0
        self.initial_sol = np.zeros(int(solution_dimension))

        self.gaussianSettings = gaussianEmitterSettings
        self.isolineSettings = isolineEmitterSettings
        self.optimizingSettings = optimizingEmitterSettings
        self.improvementSettings = improvementEmitterSettings
        self.randomDirectionSettings = randomDirectionEmitterSettings


           

