import numpy as np
from typing import Tuple


ma_mutation_template = """
import numpy as np
from typing import Tuple

def ma_mutation(solution: np.ndarray, n_vars: int) -> np.ndarray:
    mutated = solution.copy()
    pos1, pos2 = np.random.choice(n_vars, 2, replace=False)
    mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
    return mutated
"""


def ma_mutation(solution: np.ndarray, n_vars: int) -> np.ndarray:
    mutated = solution.copy()
    pos1, pos2 = np.random.choice(n_vars, 2, replace=False)
    mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
    return mutated


op_crossover_template = """ 
import numpy as np
from typing import Tuple

def op_crossover(
    parent1: np.ndarray, parent2: np.ndarray, n_vars: int
) -> Tuple[np.ndarray, np.ndarray]:
   
    r1 = np.linalg.norm(parent1)
    r2 = np.linalg.norm(parent2)
    theta1 = np.arccos(parent1[0] / r1) if r1 > 0 else 0
    theta2 = np.arccos(parent2[0] / r2) if r2 > 0 else 0

    # Blend radii
    alpha = np.random.uniform(0.3, 0.7)
    r_child1 = alpha * r1 + (1 - alpha) * r2
    r_child2 = (1 - alpha) * r1 + alpha * r2

    # Interpolate angles randomly
    beta = np.random.uniform(0, 1, n_vars - 1)
    angles_child1 = theta1 * beta + theta2 * (1 - beta)
    angles_child2 = theta2 * beta + theta1 * (1 - beta)

    # Convert back to cartesian coordinates
    def sph2cart(r, angles):
        sin_angles = np.sin(angles)
        cos_angles = np.cos(angles)
        x = np.zeros(n_vars)
        x[0] = r * cos_angles[0] if len(angles) > 0 else r
        for i in range(1, n_vars):
            x[i] = (
                r * np.prod(sin_angles[:i]) * (cos_angles[i] if i < len(angles) else 1)
            )
        return x

    child1 = sph2cart(r_child1, angles_child1)
    child2 = sph2cart(r_child2, angles_child2)

    return np.clip(child1, 0, 1), np.clip(child2, 0, 1)

"""


def op_crossover(
    parent1: np.ndarray, parent2: np.ndarray, n_vars: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced single-point crossover for continuous real-valued vectors optimized for HV.

    Performs genetic crossover with focus on preserving/maximizing hypervolume:
    1. Maintains solution feasibility through clipping
    2. Preserves solution diversity through random crossover points
    3. Handles edge cases robustly while maintaining vector characteristics

    Args:
        parent1: First parent solution vector (n_vars-dimensional)
        parent2: Second parent solution vector (n_vars-dimensional)
        n_vars: Expected length of solution vectors

    Improvement Suggestions:
        - Implement adaptive crossover probability based on population diversity
        - Add SBX (Simulated Binary Crossover) for better HV preservation
        - Consider blend crossover (BLX-α) for improved exploration

    Returns:
        Two child vectors of same length as parents
    """
    r1 = np.linalg.norm(parent1)
    r2 = np.linalg.norm(parent2)
    theta1 = np.arccos(parent1[0] / r1) if r1 > 0 else 0
    theta2 = np.arccos(parent2[0] / r2) if r2 > 0 else 0

    # Blend radii
    alpha = np.random.uniform(0.3, 0.7)
    r_child1 = alpha * r1 + (1 - alpha) * r2
    r_child2 = (1 - alpha) * r1 + alpha * r2

    # Interpolate angles randomly
    beta = np.random.uniform(0, 1, n_vars - 1)
    angles_child1 = theta1 * beta + theta2 * (1 - beta)
    angles_child2 = theta2 * beta + theta1 * (1 - beta)

    # Convert back to cartesian coordinates
    def sph2cart(r, angles):
        sin_angles = np.sin(angles)
        cos_angles = np.cos(angles)
        x = np.zeros(n_vars)
        x[0] = r * cos_angles[0] if len(angles) > 0 else r
        for i in range(1, n_vars):
            x[i] = (
                r * np.prod(sin_angles[:i]) * (cos_angles[i] if i < len(angles) else 1)
            )
        return x

    child1 = sph2cart(r_child1, angles_child1)
    child2 = sph2cart(r_child2, angles_child2)

    return np.clip(child1, 0, 1), np.clip(child2, 0, 1)


op_mutation_template = """ 
import numpy as np
from typing import Tuple

def op_mutation(solution: np.ndarray, n_vars: int) -> np.ndarray:
    
    try:
        mutated = solution.copy()
        idx = np.random.randint(0, n_vars)
        mutated[idx] += np.random.normal(0, 1)
        return mutated
    except Exception:
        return solution.copy()
"""


def op_mutation(solution: np.ndarray, n_vars: int) -> np.ndarray:
    """
    Process priority mutation operator for real-valued vectors

    Args:
        solution: Process priority chromosome (n_vars-dimensional)
        n_vars: Chromosome length

    Notes:
        1. Input solution is a 1D numpy.ndarray (length = n_vars)
        2. Avoid list operations (pop/append/remove/insert)
        3. Return value must be numpy.ndarray with same length
        4. Return input copy on error to ensure evolution continuity

    Returns:
        Mutated chromosome (same length as input)
    """
    try:
        mutated = solution.copy()
        idx = np.random.randint(0, n_vars)
        mutated[idx] += np.random.normal(0, 1)
        return mutated
    except Exception:
        return solution.copy()


ma_crossover_template = """ 
import numpy as np
from typing import Tuple

def ma_crossover(
    parent1: np.ndarray, parent2: np.ndarray, n_vars: int
) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.random.randint(0, 2, size=n_vars)
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2
"""


def ma_crossover(
    parent1: np.ndarray, parent2: np.ndarray, n_vars: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniform crossover operator for machine assignment chromosomes

    Args:
        parent1: First parent chromosome (n_vars-dimensional)
        parent2: Second parent chromosome (n_vars-dimensional)
        n_vars: Chromosome length

    Returns:
        Two child chromosomes generated by uniform crossover
    """
    mask = np.random.randint(0, 2, size=n_vars)
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2
