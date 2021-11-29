import numpy as np

def normalize_vector(vector : np.ndarray) -> np.ndarray:
    norm = np.sum(np.power(vector,2))
    norm = pow(norm, 0.5)
    return norm