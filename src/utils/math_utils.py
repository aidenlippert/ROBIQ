import numpy as np

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two 3D points.
    Parameters:
    - point1, point2: Tuples or arrays representing the (x, y, z) coordinates.
    Returns:
    - distance: The Euclidean distance between the two points.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def normalize_vector(vector):
    """
    Normalize a vector to have a magnitude of 1.
    Parameters:
    - vector: A numpy array representing the vector to normalize.
    Returns:
    - normalized_vector: The normalized vector.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm