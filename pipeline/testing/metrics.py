import ot
import numpy as np
from typing import Any

# Evaluation metric:
    # Histogram error: 
    #   H(S) = {(S_i, c_i) | i = 1,...,N}, where S is the statistic, S_i is the i-th summary stat bin center, c_i is the count
    #   Err(H_true, H_pred) = sum_i ||c_true_i - c_pred_i||_1
    # Energy spectrum error:
    #  U_n = (u(t_i)_n, G(u(t_i))_n, G(G(u(t_i)))_n, ...), where n is the trajectory sample index, G is the emulator operator
    #  ESE = mean_n || FFT(U_n_true) - FFT(U_n_pred)||_1 / ||FFT(U_n_true)||_1

def get_IPM(data_true: np.ndarray, data_pred: np.ndarray, p: int = 2) -> float:
    """Compute the Integral Probability Metric (IPM) between true and predicted data.

    Args:
        data_true (np.ndarray): True data samples of shape (batch_size, n_samples, n_dims).
        data_pred (np.ndarray): Predicted data samples of shape (batch_size, n_samples, n_dims).
        p (int, optional): The order of the norm. Defaults to 2.

    Returns:
        ipm (float): The computed IPM value.
    """
    if data_true.shape != data_pred.shape:
        raise ValueError(f"Shape mismatch: {data_true.shape} vs {data_pred.shape}")

    batch_size = data_true.shape[0]
    ipm_values = []

    for b in range(batch_size):
        M = ot.dist(data_true[b], data_pred[b], metric='euclidean')  # Cost matrix
        a = np.ones((data_true.shape[1],)) / data_true.shape[1]  # Uniform distribution
        b_dist = np.ones((data_pred.shape[1],)) / data_pred.shape[1]  # Uniform distribution
        W = ot.emd(a, b_dist, M)  # Optimal transport plan
        ipm_b = np.sum(W * M ** p) ** (1 / p)
        ipm_values.append(ipm_b)

    return float(np.mean(ipm_values))

def get_histogram(data: np.ndarray, num_bins: int, density: bool = True) -> list[tuple[np.ndarray[tuple[int, int], np.dtype[Any]], np.ndarray[tuple[int, int], np.dtype[Any]]]]:
    """Compute histogram of the data.

    Args:
        data (np.ndarray): Input data array of shape:
            - (batch_size, n_samples, n_dims) for batched n-D
        num_bins (int): Number of histogram bins per dimension.
        density (bool, optional): If True, normalize the histogram. Defaults to True.

    Returns:
        histograms (list[tuple]): List of length batch_size, each element is a tuple of: 
            - Bin centers (np.ndarray): Shape (n_features, num_bins) -  Histogram bin centers for each dimension.
            - counts (np.ndarray): Shape (n_features, num_bins) -  Histogram counts for each dimension.
    """
    
    if data.ndim != 3:
        raise ValueError(f"Expected 3D input (batch_size, n_samples, n_dims), got {data.shape}")

    batch_size, _, n_dims = data.shape

    histograms = []
    
    for b in range(batch_size):
        bin_centers = np.zeros((n_dims, num_bins))
        counts = np.zeros((n_dims, num_bins))
        for dim in range(n_dims):
            
            hist_counts, bin_edges = np.histogram(data[b, :, dim], bins=num_bins, density=density)
            counts[dim] = hist_counts
            bin_centers[dim] = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        histograms.append((bin_centers, counts))

    return histograms

def compute_histogram_error(H_true: list[tuple[np.ndarray[int, np.dtype], np.ndarray[int, np.dtype]]], 
                            H_hat: list[tuple[np.ndarray[int, np.dtype], np.ndarray[int, np.dtype]]], 
                            aggregate: str = 'mean') -> float:
    """Compute histogram error between true and predicted summary statistics.

    Args:
        H_true (list[tuple[np.ndarray, np.ndarray]]): List of true histograms for each sample.
        H_hat (list[tuple[np.ndarray, np.ndarray]]): List of predicted histograms for each sample.
        aggregate (str, optional): Method to aggregate errors ('mean' or 'sum'). Defaults to 'mean'.

    Returns:
        histogram_error (float): Histogram error value.
    """
    if len(H_true) != len(H_hat):
        raise ValueError(f"Batch size mismatch: {len(H_true)} vs {len(H_hat)}")
    
    true_counts = np.stack([counts for _, counts in H_true])  # (batch_size, n_dims, num_bins)
    hat_counts = np.stack([counts for _, counts in H_hat])    # (batch_size, n_dims, num_bins)
    
    # L1 distance for each feature dimension
    hist_error_per_dim = np.sum(np.abs(hat_counts - true_counts), axis=2)

    if aggregate == 'mean':
        return np.mean(hist_error_per_dim)
    else:
        raise ValueError(f"Unknown aggregate method: {aggregate}.")

def compute_energy_spectrum_error(u_true: np.ndarray, u_hat: np.ndarray, eps: float = 1e-8) -> float:
    """Energy spectrum error (ESE) between true and predicted trajectories.
        ESE = mean_n || |FFT(U_n_true)|**2 - |FFT(U_n_pred)|**2 ||_1 / || |FFT(U_n_true)|**2| |_1

    Args:
        u_true (np.ndarray): True trajectory samples.
        u_hat (np.ndarray): Predicted trajectory samples.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-8.

    Returns:
        float: Energy spectrum error (ESE) value.
    """
    U = np.fft.rfft(u_true, axis=1)
    U_hat = np.fft.rfft(u_hat, axis=1)
    P_true = np.abs(U) ** 2
    P_hat = np.abs(U_hat) ** 2

    ESE_per_sample = np.mean(np.abs(P_true - P_hat), axis=1) / (np.mean(P_true, axis=1) + eps)
    ESE = np.mean(ESE_per_sample).item()

    return ESE

def compute_trajectory_errors(u_true, u_hat) -> dict[str, Any]:
    """
    Compute various error metrics for trajectory predictions.
    
    Args:
        u_true: True trajectory tensor, shape [batch, T, d]
        u_hat: Predicted trajectory tensor, shape [batch, T, d]
    
    Returns:
        Dictionary of error metrics
    """
    # Computing MSE, MAE, RMSE
    mse = np.mean((u_true - u_hat) ** 2).item()
    mse_per_time = np.mean((u_true - u_hat) ** 2, axis=(0, 2))  # [T]
    mse_per_component = np.mean((u_true - u_hat) ** 2, axis=(0, 1))  # [d]

    mae = np.mean(np.abs(u_true - u_hat)).item()
    mae_per_time = np.mean(np.abs(u_true - u_hat), axis=(0, 2))  # [T]
    mae_per_component = np.mean(np.abs(u_true - u_hat), axis=(0, 1))  # [d]

    rmse = np.sqrt(np.mean((u_true - u_hat) ** 2)).item()
    rmse_per_time = np.sqrt(np.mean((u_true - u_hat) ** 2, axis=(0, 2)))  # [T]
    rmse_per_component = np.sqrt(np.mean((u_true - u_hat) ** 2, axis=(0, 1)))  # [d]

    # Computing lp (l1, l2 and l-infinity) errors
    l1_error = np.mean(np.abs(u_true - u_hat)).item()
    l2_error = np.mean(np.linalg.norm(u_true - u_hat, axis=-1) / (np.linalg.norm(u_true, axis=-1) + 1e-8)).item()
    l_infinity_error = np.max(np.abs(u_true - u_hat)).item()

    # Computing energy spectrum error
    energy_spectrum_error = compute_energy_spectrum_error(u_true, u_hat)
    
    # Computing histogram error
    k = int(np.ceil(np.sqrt(np.max([u_true.shape[1], u_hat.shape[1]])))) # Sqrt rule for defining the number of bins
    H_true = get_histogram(u_true, num_bins=k, density=True)
    H_hat = get_histogram(u_hat, num_bins=k, density=True)
    hist_error = compute_histogram_error(H_true, H_hat)
    emd = get_IPM(u_true, u_hat, p=2)


    return {
        'mse': mse,
        'mse_per_time': mse_per_time,
        'mse_per_component': mse_per_component,
        'mae': mae,
        'mae_per_time': mae_per_time,
        'mae_per_component': mae_per_component,
        'rmse': rmse,
        'rmse_per_time': rmse_per_time,
        'rmse_per_component': rmse_per_component,
        'l1_error': l1_error,
        'l2_error': l2_error,
        'l_infinity_error': l_infinity_error,
        'energy_spectrum': energy_spectrum_error,
        'histogram_error': hist_error,
        'ipm': emd
    }

def compute_summary_errors(s_true, s_hat) -> dict[str, Any]:
    """
    Compute error metrics for summary statistics.
    
    Args:
        s_true: True summary statistics tensor
        s_hat: Predicted summary statistics tensor
    
    Returns:
        Dictionary of error metrics
    """
    # Computing MSE, MAE, RMSE
    mse = np.mean((s_true - s_hat) ** 2).item()
    mse_per_time = np.mean((s_true - s_hat) ** 2, axis=(0, 2))  # [T]
    mse_per_component = np.mean((s_true - s_hat) ** 2, axis=(0, 1))  # [n]

    mae = np.mean(np.abs(s_true - s_hat)).item()
    mae_per_time = np.mean(np.abs(s_true - s_hat), axis=(0, 2))  # [T]
    mae_per_component = np.mean(np.abs(s_true - s_hat), axis=(0, 1))  # [n]

    rmse = np.sqrt(np.mean((s_true - s_hat) ** 2)).item()
    rmse_per_time = np.sqrt(np.mean((s_true - s_hat) ** 2, axis=(0, 2)))  # [T]
    rmse_per_component = np.sqrt(np.mean((s_true - s_hat) ** 2, axis=(0, 1)))  # [n]

    # Computing lp (l1, l2 and l-infinity) errors
    l1_error = np.mean(np.abs(s_true - s_hat)).item()
    l2_error = np.mean(np.linalg.norm(s_true - s_hat, axis=-1) / (np.linalg.norm(s_true, axis=-1) + 1e-8)).item()
    l_infinity_error = np.max(np.abs(s_true - s_hat)).item()

    # Computing energy spectrum error
    energy_spectrum_error = compute_energy_spectrum_error(s_true, s_hat)

    # Computing histogram error
    k = int(np.ceil(np.sqrt(np.max([s_true.shape[1], s_hat.shape[1]])))) # Sqrt rule for defining the number of bins
    H_true = get_histogram(s_true, num_bins=k, density=True)
    H_hat = get_histogram(s_hat, num_bins=k, density=True)
    hist_error = compute_histogram_error(H_true, H_hat)

    return {
        'mse': mse,
        'mse_per_time': mse_per_time,
        'mse_per_component': mse_per_component,
        'mae': mae,
        'mae_per_time': mae_per_time,
        'mae_per_component': mae_per_component,
        'rmse': rmse,
        'rmse_per_time': rmse_per_time,
        'rmse_per_component': rmse_per_component,
        'l1_error': l1_error,
        'l2_error': l2_error,
        'l_infinity_error': l_infinity_error,
        'energy_spectrum': energy_spectrum_error,
        'histogram_error': hist_error,
    }
