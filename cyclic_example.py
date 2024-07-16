from data_loader.sample_dcg import sample_er_dcg
from data_loader.sample_data import sample_param_unif, sample_data

from local_isa_ling.local_isa_ling import local_isa_ling
from utils.markov_blanket import get_true_markov_blanket, estimate_markov_blanket
from utils.metrics import count_accuracy, count_accuracy_of_mb
from utils.utils import set_random_seed, map_local_structure


if __name__ == '__main__':
    num_vars, degree = 50, 4
    num_samples = 2000
    # Generate data
    set_random_seed(6)
    B_support = sample_er_dcg(num_vars, degree, max_cycle=10)
    set_random_seed(6)
    B, noise_scales = sample_param_unif(B_support)
    set_random_seed(6)
    X = sample_data(B, noise_scales, num_samples)
    B_support, B = B_support.T, B.T    # Transpose to make each row correspond to parents
    target = 26    # Pick target for local causal discovery
    # Estimate Markov blanket and local structure
    set_random_seed(1)
    mb_est = estimate_markov_blanket(X, target)
    mb_true = get_true_markov_blanket(B, target)
    results_mb = count_accuracy_of_mb(mb_true, mb_est)
    print("Accuracy of estimated Markov blanket:", results_mb)
    local_indices = [target] + list(mb_est)
    X_local = X[:, local_indices]
    set_random_seed(1)
    params = local_isa_ling(X_local, target=0, postprocess_type='block_diagonal')
    # Map local structure back to global indices
    dcg_est_mapped = map_local_structure(params['dcg'], local_indices, num_vars)
    results = count_accuracy(B, dcg_est_mapped, target)
    print("Accuracy of estimated local structure:", results)