from typing import List

from knapsack import KnapsackProblem
from eagle import GoldenEagleOptimizer

from transfer_functions import q_functions
from binarization_functions import binarization_functions

import numpy as np
import matplotlib.pyplot as plt

iterations : int = 50
population_size : int = 100
attack_propensity_start : float =1
attack_propensity_end : float =2
cruise_propensity_start : float =1
cruise_propensity_end : float =0.5
lower_limit : float =-1
upper_limit : float =1
tau : int = 5 # Tournament size

transfer_functions : List[str]  = [
    "q1", 
    "q2"
]

binarizations : List[str] = [
    "bb",
    "btlboe", 
    "btlboer",
    "btlboerw", 
    "btlboet", 
]

files : List[str] = [
    "instances/large_scale/knapPI_1_100_1000_1",
    "instances/large_scale/knapPI_1_200_1000_1",
    "instances/large_scale/knapPI_1_500_1000_1",
    "instances/large_scale/knapPI_1_1000_1000_1",
    "instances/large_scale/knapPI_1_2000_1000_1",
    "instances/large_scale/knapPI_2_100_1000_1",
    "instances/large_scale/knapPI_2_200_1000_1",
    "instances/large_scale/knapPI_2_500_1000_1",
    "instances/large_scale/knapPI_2_1000_1000_1",
    "instances/large_scale/knapPI_2_2000_1000_1",
    "instances/large_scale/knapPI_3_100_1000_1",
    "instances/large_scale/knapPI_3_200_1000_1",
    "instances/large_scale/knapPI_3_500_1000_1",
    "instances/large_scale/knapPI_3_1000_1000_1",
    "instances/large_scale/knapPI_3_2000_1000_1",
    "instances/low-dimensional/f1_l-d_kp_10_269",
    "instances/low-dimensional/f2_l-d_kp_20_878",
    "instances/low-dimensional/f3_l-d_kp_4_20",
    "instances/low-dimensional/f4_l-d_kp_4_11",
    "instances/low-dimensional/f5_l-d_kp_15_375",
    "instances/low-dimensional/f6_l-d_kp_10_60",
    "instances/low-dimensional/f7_l-d_kp_7_50",
    "instances/low-dimensional/f8_l-d_kp_23_10000",
    "instances/low-dimensional/f9_l-d_kp_5_80",
    "instances/low-dimensional/f10_l-d_kp_20_879"
]

for archivo in files:
    _file = open(f'{archivo}_results.csv','w+')
    for binarization in binarizations:
        for transfer in transfer_functions:
            row = []
            instance = KnapsackProblem(archivo)

            optimizer = GoldenEagleOptimizer(
                iterations=iterations,
                population_size=population_size,
                attack_propensity_start=attack_propensity_start,
                attack_propensity_end=attack_propensity_end,
                cruise_propensity_start=cruise_propensity_start,
                cruise_propensity_end=cruise_propensity_end,
                num_variables=instance.num_elements,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                problem_instance=instance,
                transfer_function=q_functions[transfer],
                binarization_function=binarization_functions[binarization],
                tau=tau
            )
            row.append(f"{transfer}_{binarization}")
    
            for i in range(35):
                fitness, solution, convergence = optimizer.solve()
                #fitnesses.append(fitness)
                #curves.append(convergence)
                print(fitness)
                row.append(str(fitness))
            _file.write(','.join(row)+"\n")
            print(f"{archivo} {transfer} {binarization} : {fitness}")
    _file.close()
