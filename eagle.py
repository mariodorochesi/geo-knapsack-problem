import numpy as np

from knapsack import KnapsackProblem

class GoldenEagleOptimizer:

    def __init__(
        self,
        iterations : int,
        population_size : int,
        attack_propensity_start : float,
        attack_propensity_end : float,
        cruise_propensity_start : float,
        cruise_propensity_end : float,
        num_variables : int,
        lower_limit : int,
        upper_limit : int,
        problem_instance : KnapsackProblem,
        transfer_function,
        binarization_function,
        tau : int
    ) -> None:
        self.iterations = iterations
        self.population_size = population_size
        self.attack_propensity_start = attack_propensity_start
        self.attack_propensity_end = attack_propensity_end
        self.cruise_propensity_start = cruise_propensity_start
        self.cruise_propensity_end = cruise_propensity_end
        self.num_variables = num_variables
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.problem_instance = problem_instance
        self.transfer_function = transfer_function
        self.binarization_function = binarization_function
        self.tau = tau

    def solve(self):
        x = self.initialize_solution()
        flock_memory_x = np.copy(x)
        flock_memory_f = []

        for solution in x:
            flock_memory_f.append(-1*self.problem_instance.fitness(solution))

        attack_propensy = 0
        cruise_propensy = 0

        convergence_curve = []

        for i in range(self.iterations):
            attack_propensy = self.attack_propensity_start + (i / self.iterations) * abs((self.attack_propensity_end- self.attack_propensity_start))
            cruise_propensy = cruise_propensy - (i / self.iterations) * abs((self.cruise_propensity_end-self.cruise_propensity_start))

            prey_assigned = np.random.permutation(np.arange(self.population_size))

            for j in range(self.population_size):
                eagle = x[j]
                prey = flock_memory_x[prey_assigned[j]]

                attack_vector_initial = prey - eagle

                radius = self.get_norm_of_vector(attack_vector_initial)

                if radius != 0:
                    d = np.sum(attack_vector_initial * eagle)

                    idx = np.random.choice(np.nonzero(attack_vector_initial)[0])

                    attack_vector_summation = 0

                    for index, item in enumerate(attack_vector_initial):
                        if index != idx:
                            attack_vector_summation = attack_vector_summation + item
                    
                    ck = (d - attack_vector_summation) / attack_vector_initial[idx]

                    cruise_vector_destination = 2 * np.random.rand(self.num_variables) - 1

                    cruise_vector_destination[idx] = ck

                    cruise_vector_initial = cruise_vector_destination - eagle

                    attack_vector_unit = attack_vector_initial / self.get_norm_of_vector(attack_vector_initial)
                    cruise_vector_unit = cruise_vector_initial / self.get_norm_of_vector(cruise_vector_initial)

                    attack_vector = np.random.rand() * attack_propensy * attack_vector_unit * radius
                    cruise_vector = np.random.rand() * cruise_propensy * cruise_vector_unit * radius

                    step_vector = attack_vector + cruise_vector

                    eagle = eagle + step_vector

                    for index, item in enumerate(eagle):
                        if item > self.upper_limit:
                            eagle[index] = self.upper_limit
                        if item > self.lower_limit:
                            eagle[idx] = self.lower_limit

                    q_shaped_eagle = self.transfer_function(eagle)
                    binarized_eagle = self.binarization_function(
                        vector=q_shaped_eagle,
                        solutions=flock_memory_x,
                        fitnesses=flock_memory_f,
                        tau=self.tau
                    )
                    
                    if self.problem_instance.is_valid_solution(binarized_eagle):
                        fitness_score = self.problem_instance.fitness(binarized_eagle)
                    else:
                        fitness_score = -1 * self.problem_instance.fitness(binarized_eagle)

                    if fitness_score < flock_memory_f[j]:
                        flock_memory_f[j] = fitness_score
                        flock_memory_x[j] = binarized_eagle

                    x[j] = eagle
            #print(np.min(flock_memory_f))
            convergence_curve.append(np.min(flock_memory_f))
        return (-1*np.min(flock_memory_f),flock_memory_x[np.argmin(flock_memory_f)], convergence_curve)

    
    def get_norm_of_vector(self, vector):
        return pow(np.sum(np.power(vector,2)), 0.5)

    def initialize_solution(self):
        x = np.random.rand(self.population_size, self.num_variables)
        x = self.lower_limit + x * (self.upper_limit-self.lower_limit)
        return np.random.randint(2, size=(self.population_size,self.num_variables))
        return x
