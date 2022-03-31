
from pathlib import Path
import numpy as np


class ExperimentSettings():
    def __init__(self, alg, population_size, generations, solution_dimension):
        self.alg = alg
        self.population_size = population_size
        self.generations = generations
        self.solution_dimension = solution_dimension    
        self.metrics = {
                "QD Score": {"x": [0],"y": [0.0],},
                "Archive Coverage": {"x": [0],"y": [0.0],},}
        self.name = f"{str(self.alg)}_{str(self.solution_dimension)}"

class ArchiveSettings():
    def __init__(self,  gridTypeClass, gridDimensions , max_bound, seed=None): 
        self.gridTypeClass = gridTypeClass
        self.gridDimensions = gridDimensions
        self.max_bound = float(max_bound)
        self.gridBounds = [(-self.max_bound, self.max_bound), (-self.max_bound, self.max_bound)]
        self.seed = seed

class LoggingSettings():
    def __init__(self, outdir,log_freq, seed_log, non_logging_time):
        self.outdir = outdir
        self.log_freq = log_freq
        self.seed = seed_log
        self.non_logging_time = non_logging_time

        self.outdir = Path(self.outdir)
        if not self.outdir.is_dir():
            self.outdir.mkdir()  
        



class EmittersSettings():
    #emitter_settings = EmittersSettings(, set_total_emitter_counts, make_emitter_seeds, dictionary['seed-emitters'])

    #dictionary['gaussian'],dictionary['improvement'],dictionary['iso_line'],dictionary['optimizing'],
    # dictionary['random_direction'], set_total_emitter_counts, make_emitter_seeds, 
    # dictionary['solution_dimension'], dictionary['seed-emitters'], dictionary['sigma0'],
    # dictionary['gaussian_pop_size'], dictionary['isoline_pop_size'], dictionary['iso_sigma'], dictionary['line_sigma'],
    # dictionary['rd_selection_rule'], dictionary['rd_restart_rule'], dictionary['rd_weight_rule'], 
    # dictionary['rd_pop_size'], 
    # dictionary['imp_selection_rule'], dictionary['imp_restart_rule'], dictionary['imp_weight_rule'],
    # dictionary['imp_pop_size'], dictionary['opt_selection_rule'], dictionary['opt_restart_rule'], dictionary['opt_weight_rule'],
    # dictionary['opt_pop_size'])
    def __init__(
        self, gaussian, improvement, iso_line, optimizing, random_direction, set_total_emitter_counts, make_emitter_seeds, 
        solution_dimension, sigma0, gaussian_pop_size, isoline_pop_size, iso_sigma, line_sigma, rd_selection_rule, rd_restart_rule, rd_weight_rule, rd_pop_size, imp_selection_rule, imp_restart_rule, imp_weight_rule, imp_pop_size, opt_selection_rule, opt_restart_rule, opt_weight_rule, opt_pop_size, 
        seed = None):
        
   
        self.total_num_emitters = set_total_emitter_counts(gaussian, improvement, iso_line, optimizing, random_direction)
        self.seed = seed
        self.emitter_seeds = make_emitter_seeds(seed, self.total_num_emitters)
        
        self.gaussian = gaussian
        self.improvement = improvement
        self.iso_line = iso_line
        self.optimizing = optimizing
        self.random_direction = random_direction

        self.solution_dimension = solution_dimension

        self.sigma0 = sigma0
        self.gaussian_pop_size = gaussian_pop_size
        self.isoline_pop_size = isoline_pop_size
        self.iso_sigma = iso_sigma
        self.line_sigma = line_sigma
        self.rd_selection_rule = rd_selection_rule
        self.rd_restart_rule = rd_restart_rule
        self.rd_weight_rule = rd_weight_rule 
        self.rd_pop_size = rd_pop_size
        self.imp_selection_rule = imp_selection_rule
        self.imp_restart_rule = imp_restart_rule
        self.imp_weight_rule = imp_weight_rule
        self.imp_pop_size = imp_pop_size
        self.opt_selection_rule = opt_selection_rule
        self.opt_restart_rule = opt_restart_rule
        self.opt_weight_rule = opt_weight_rule
        self.opt_pop_size = opt_pop_size
        
       
        self.initial_sol = np.zeros(int(solution_dimension))


           

