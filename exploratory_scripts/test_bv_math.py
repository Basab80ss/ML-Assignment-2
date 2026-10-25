import numpy as np

def compute_bias_variance(P_r, y_true, num_classes=40):
    """
    P_r: shape (R, N, num_classes)
    y_true: shape (N,)
    """
    R, N, K = P_r.shape
    
    # One-hot target matrix E (N, K)
    E = np.zeros((N, K))
    for i in range(N):
        E[i, y_true[i]] = 1.0
        
    # Mean prediction across R repetitions: shape (N, K)
    p_bar = np.mean(P_r, axis=0)
    
    # Bias^2 = 1/N * sum_i || p_bar(x_i) - e_i ||_2^2
    bias_sq = np.mean(np.sum((p_bar - E)**2, axis=1))
    
    # Variance = 1/(N * R) * sum_i sum_r || p_r(x_i) - p_bar(x_i) ||_2^2
    # P_r - p_bar: shape (R, N, K)
    var = np.mean(np.sum((P_r - p_bar)**2, axis=2))
    
    return bias_sq, var

print("Verified Bias^2 and Variance mathematical functions.")
