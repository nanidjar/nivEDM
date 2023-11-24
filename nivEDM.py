import numpy as np
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors

def predict_next_point(dmatrix, point, num_nearest_points, distance_scale):
    """
    Predict the next point in a time series using nearest neighbors and linear regression.

    :param dmatrix: The data matrix containing time series data with occasional NaN values.
    :param point: The reference point for nearest neighbors prediction.
    :param num_nearest_points: Number of nearest neighbors to consider.
    :param distance_scale: Scale factor for distance in weighting.
    :return: Predicted next point in the time series.
    """
    # Remove NaNs and update time steps
    valid_rows = ~np.isnan(dmatrix).any(axis=1)
    dmatrix_clean = dmatrix[valid_rows]
    time_steps = np.arange(len(dmatrix))[valid_rows]

    # Calculate differences in time steps
    time_diffs = np.diff(time_steps, append=0)

    # Find the nearest points to the given point
    nbrs = NearestNeighbors(n_neighbors=num_nearest_points + 1)
    nbrs.fit(dmatrix_clean[:-1])  # Exclude the last point, as it has no subsequent point
    distances, indices = nbrs.kneighbors([point])
    distances, indices = distances[0][1:], indices[0][1:]  # Exclude the point itself if present

    # Filter neighbors with consecutive time steps
    consecutive_neighbors = time_diffs[indices] == 1
    distances, indices = distances[consecutive_neighbors], indices[consecutive_neighbors]

    # Check if there are enough neighbors after filtering
    if len(indices) < num_nearest_points:
        raise ValueError("Not enough neighbors with consecutive time steps. Try reducing num_nearest_points.")

    # Apply weighting based on distances
    weights = np.exp(-distances * distance_scale) / np.max(np.exp(-distances * distance_scale))

    # Weight the nearest points
    nearest_points = dmatrix_clean[indices]
    weighted_nearest_points = nearest_points * weights[:, np.newaxis]

    # Prepare and weight the target points (the subsequent points in time)
    target_points = dmatrix_clean[indices + 1]
    weighted_target_points = target_points * weights[:, np.newaxis]

    # Fit a Ridge regression model
    model = linear_model.Ridge(alpha=0.1)
    model.fit(weighted_nearest_points, weighted_target_points)

    # Predict the next point
    predicted_next_point = model.predict([point])[0]

    return predicted_next_point


def predict_first(dmatrix, point, num_nearest_points, distance_scale, time_steps_ahead):
    """
    Predict the value of the first column in a time series a specified number of time steps ahead 
    based on the current state of all other columns. Returns NaN if no valid nearest neighbors are found.

    :param dmatrix: The data matrix containing time series data with occasional NaN values.
    :param point: The reference point (excluding the first column) for nearest neighbors prediction.
    :param num_nearest_points: Number of nearest neighbors to consider.
    :param distance_scale: Scale factor for distance in weighting.
    :param time_steps_ahead: Number of time steps ahead for prediction.
    :return: Predicted value of the first column in the time series for the specified number of time steps ahead, or NaN if no valid neighbors.
    """
    # Remove NaNs and update time steps
    valid_rows = ~np.isnan(dmatrix).any(axis=1)
    dmatrix_clean = dmatrix[valid_rows]
    time_steps = np.arange(len(dmatrix))[valid_rows]
    
    # Find the nearest points to the given point (excluding the first column)
    nbrs = NearestNeighbors(n_neighbors=num_nearest_points)
    nbrs.fit(dmatrix_clean[:, 1:])
    distances, indices = nbrs.kneighbors([point[1:]])
    
    # Adjust future indices based on actual time steps
    valid_future_indices = []
    valid_distances = []
    valid_indices = []
    for dist, idx in zip(distances[0], indices[0]):
        current_time_step = time_steps[idx]
        future_time_step = current_time_step + time_steps_ahead
        future_idx = np.where(time_steps >= future_time_step)[0][0] if np.any(time_steps >= future_time_step) else -1
        if future_idx != -1:
            valid_future_indices.append(future_idx)
            valid_distances.append(dist)
            valid_indices.append(idx)
    
    # Check if valid nearest neighbors are found
    if not valid_future_indices:
        return np.nan
    
    # Apply inverse exponential distance weighting
    weights = np.exp(-np.array(valid_distances) * distance_scale)

    # Weight the nearest points and the corresponding future values of the first column
    nearest_points = dmatrix_clean[valid_indices, 1:]
    weighted_nearest_points = nearest_points * weights[:, np.newaxis]
    
    future_values_first_column = dmatrix_clean[valid_future_indices, 0]
    weighted_future_values = future_values_first_column * weights

    # Fit a Ridge regression model
    model = linear_model.Ridge(alpha=0.1)
    model.fit(weighted_nearest_points, weighted_future_values)

    # Predict the future value of the first column
    predicted_future_value = model.predict([point[1:]])[0]

    return predicted_future_value