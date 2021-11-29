from typing import Optional

import numpy as np

def basic_binarization(vector, solutions = None, fitnesses = None, tau = None):
    result = np.zeros(vector.shape)
    random = np.random.rand()
    for i in range(len(vector)):
        if random <= vector[i]:
            result[i] = 1
        else:
            result[i] = 0
    return result

def btlboe(vector, solutions = None, fitnesses = None, tau = None):
    result = np.zeros(vector.shape)
    solution = solutions[np.argmin(fitnesses)]
    for i in range(len(vector)):
        random = np.random.rand()
        if random <= vector[i]:
            result[i] = (solution[i] * -1) + 1
        else:
            result[i] = solution[i]
    return result

def btlboerw(vector, solutions = None, fitnesses = None, tau = None):
    total_fit = np.sum(np.absolute(fitnesses))
    fit_probability = np.zeros(len(fitnesses))

    for i in range(len(fit_probability)):
        fit_probability[i] = abs(fitnesses[i])/total_fit
    
    idx = np.random.choice(list(range(0,solutions.shape[0])), p=fit_probability)

    solution = solutions[idx]
    result = np.zeros(vector.shape)
    random = np.random.rand()
    for i in range(len(vector)):
        if random <= vector[i]:
            result[i] = (solution[i] * -1) + 1
        else:
            result[i] = solution[i]
    return result

def btlboet(vector, solutions = None, fitnesses = None, tau = None):
    idxs = [i[0] for i in sorted(enumerate(fitnesses),key=lambda x : x[1])[:tau]]
    idx = np.random.choice(idxs)
    result = np.zeros(vector.shape)
    solution = solutions[idx]
    for i in range(len(vector)):
        random = np.random.rand()
        if random <= vector[i]:
            result[i] = (solution[i] * -1) + 1
        else:
            result[i] = solution[i]
    return result

def btlboer(vector, solutions = None, fitnesses = None, tau = None):
    idxs = [i[0] for i in sorted(enumerate(fitnesses),key=lambda x : x[1])]
    ranks = list(range(len(fitnesses),0,-1))
    probs = np.zeros(len(fitnesses))
    for i in range(len(fitnesses)):
        probs[i] = (2*ranks[i]) / (ranks[0]*(ranks[0]+1))

    idx = np.random.choice(idxs, p=probs)
    result = np.zeros(vector.shape)
    solution = solutions[idx]
    for i in range(len(vector)):
        random = np.random.rand()
        if random <= vector[i]:
            result[i] = (solution[i] * -1) + 1
        else:
            result[i] = solution[i]
    return result

binarization_functions = {
    "bb" : basic_binarization,
    "btlboe" : btlboe,
    "btlboerw" : btlboerw,
    "btlboet" : btlboet,
    "btlboer" : btlboer
}