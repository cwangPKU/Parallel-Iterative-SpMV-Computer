import numpy as np
import argparse # argparse requires Python 2.7 or Python 3.2+

def load_vector_from_file(filename):
    """Loads a vector from a file (one number per line)."""
    try:
        with open(filename, 'r') as f:
            vector = [float(line.strip()) for line in f if line.strip()]
        return np.array(vector)
    except FileNotFoundError:
        print("Error: File '{}' not found.".format(filename))
        return None
    except ValueError as e:
        print("Error: Could not convert data in '{}' to float. {}".format(filename, e))
        return None
    except Exception as e: # Catch any other potential errors during file loading
        print("An unexpected error occurred while loading '{}': {}".format(filename, e))
        return None

def calculate_vector_norms(vec, vec_name="Vector"):
    """Calculates and prints L1, L2, and L-infinity norms of a vector."""
    if vec is None or len(vec) == 0:
        print("{} norms: N/A (vector is empty or None)".format(vec_name))
        return

    l1_norm = np.linalg.norm(vec, ord=1)
    l2_norm = np.linalg.norm(vec, ord=2)
    linf_norm = np.linalg.norm(vec, ord=np.inf)

    print("{} Norms:".format(vec_name))
    print("  L1 (Sum of abs values): {:.6e}".format(l1_norm))
    print("  L2 (Euclidean)        : {:.6e}".format(l2_norm))
    print("  L-inf (Max abs value) : {:.6e}".format(linf_norm))
    return l1_norm, l2_norm, linf_norm


def calculate_errors(vec1, vec2, rel_tol=1e-5, abs_tol=1e-6):
    """
    Calculates and prints various error metrics between two vectors.
    Returns True if all elements are close, False otherwise.
    """
    if vec1.shape != vec2.shape:
        print("Error: Vectors have different shapes: {} vs {}".format(vec1.shape, vec2.shape))
        return False

    print("--- Reference Vector (File 1) Stats ---")
    calculate_vector_norms(vec1, "Reference Vector")
    print("--- Vector Under Test (File 2) Stats ---") # Optional: also print norms for vec2
    calculate_vector_norms(vec2, "Test Vector")
    print("--- Error Analysis ---")


    diff = vec1 - vec2
    abs_diff = np.abs(diff)
    max_abs_err = np.max(abs_diff) # This is also the L-infinity norm of the difference vector

    print("Difference Vector (Ref - Test) Stats:")
    _, l2_norm_diff, linf_norm_diff = calculate_vector_norms(diff, "Difference Vector")
    # linf_norm_diff is equivalent to max_abs_err

    epsilon = 1e-15 
    
    relative_errors = np.zeros_like(vec1)
    non_zero_mask_vec1 = np.abs(vec1) > epsilon
    
    if np.any(non_zero_mask_vec1):
        relative_errors[non_zero_mask_vec1] = abs_diff[non_zero_mask_vec1] / np.abs(vec1[non_zero_mask_vec1])
    
    max_rel_err_calculated = 0.0
    if np.any(non_zero_mask_vec1):
        max_rel_err_calculated = np.max(relative_errors[non_zero_mask_vec1])


    print("\nVector length: {}".format(len(vec1)))
    print("Maximum Absolute Error (L-inf norm of difference): {:.6e}".format(max_abs_err)) # Same as linf_norm_diff
    # print("L2 Norm of Difference: {:.6e}".format(l2_norm_diff)) # Already printed above

    if np.any(non_zero_mask_vec1):
        print("Maximum Relative Error (where ref > {:.1e}): {:.6e}".format(epsilon, max_rel_err_calculated))
    else:
        print("Maximum Relative Error: N/A (reference vector is all zeros or near-zeros)")

    are_close = np.allclose(vec2, vec1, rtol=rel_tol, atol=abs_tol)

    print("--- Closeness Check ---")
    if are_close:
        print("SUCCESS: Vectors are close (rtol={:.1e}, atol={:.1e}).".format(rel_tol, abs_tol))
    else:
        print("FAILURE: Vectors are NOT close (rtol={:.1e}, atol={:.1e}).".format(rel_tol, abs_tol))
        
        # Find first differing element for debugging
        count_diff = 0
        max_diff_to_show = 5
        print("Showing up to {} differing elements:".format(max_diff_to_show))
        for i in range(len(vec1)):
            if not np.isclose(vec2[i], vec1[i], rtol=rel_tol, atol=abs_tol):
                count_diff += 1
                print("  Difference at index {}:".format(i))
                print("    File1 (Ref) [{}]: {:.15e}".format(i, vec1[i]))
                print("    File2       [{}]: {:.15e}".format(i, vec2[i]))
                print("    Abs Diff         : {:.6e}".format(np.abs(vec1[i] - vec2[i])))
                if np.abs(vec1[i]) > epsilon: # Avoid division by zero for rel diff display
                    print("    Rel Diff         : {:.6e}".format(np.abs(vec1[i] - vec2[i]) / np.abs(vec1[i])))
                else: # If ref is zero, rel diff isn't well-defined or is infinite if test is non-zero
                    print("    Rel Diff         : N/A (reference is near zero)")
                if count_diff >= max_diff_to_show:
                    print("  ... (more differences exist)")
                    break
        if count_diff == 0 and not are_close : # Should not happen if np.allclose is False
             print("  Note: np.allclose reported False, but no individual differences found with current display logic. This might indicate many very small differences near the tolerance boundary or an issue.")

    return are_close


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two vector files for closeness and show norms.")
    parser.add_argument("file1", help="Path to the first vector file (reference).")
    parser.add_argument("file2", help="Path to the second vector file (to compare).")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance (default: 1e-5).")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance (default: 1e-6).")

    args = parser.parse_args()

    print("Comparing File 1: '{}' (Reference)".format(args.file1))
    print("With File 2     : '{}' (Test)".format(args.file2))
    print("Tolerances      : rtol={:.1e}, atol={:.1e}\n".format(args.rtol, args.atol))


    vector1 = load_vector_from_file(args.file1)
    vector2 = load_vector_from_file(args.file2)

    if vector1 is not None and vector2 is not None:
        if len(vector1) == 0 and len(vector2) == 0:
            print("Both vectors are empty. Considered identical.")
        elif len(vector1) == 0 or len(vector2) == 0:
            print("One vector is empty, the other is not. Considered different.")
        else:
            if vector1.ndim > 1: vector1 = vector1.flatten()
            if vector2.ndim > 1: vector2 = vector2.flatten()

            if vector1.shape != vector2.shape:
                print("Error: Vectors have different shapes after flattening: {} vs {}".format(vector1.shape, vector2.shape))
            else:
                calculate_errors(vector1, vector2, rel_tol=args.rtol, abs_tol=args.atol)
    elif vector1 is None or vector2 is None:
        print("\nComparison aborted due to file loading errors.")
