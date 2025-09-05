from scipy.optimize import differential_evolution
import os
import importlib
import shutil
import time
from multiprocessing import cpu_count
import uuid
import fcntl
import getopt
import sys
from time import perf_counter, sleep
from tabulate import tabulate
import random


def objective_function():
    display_moments = False
    verbose = False
    static = True

    # these imports mus tbe done *after* the cohorts global parameter is set
    from calculate_emax import create_married_emax
    from calculate_emax import create_single_w_emax
    from calculate_emax import create_single_h_emax
    from calculate_emax import calculate_emax
    from calculate_emax import dump_married_emax, dump_single_emax
    import forward_simulation as fs

    w_emax = create_married_emax()
    h_emax = create_married_emax()
    w_s_emax = create_single_w_emax()
    h_s_emax = create_single_h_emax()
    if not static:
        calculate_emax(w_emax, h_emax, w_s_emax, h_s_emax, verbose)

    return fs.forward_simulation(w_emax, h_emax, w_s_emax, h_s_emax, verbose, display_moments)


def parallel_objective_wrapper(param_values, cohort, parameter_names, original_content, verbose):
    # Import modules in worker process
    import sys
    import cohorts
    import importlib

    # Set cohort FIRST before importing parameter modules
    cohorts.cohort = cohort

    worker_id = str(uuid.uuid4())[:8]
    original_param_file = f"input/parameters{cohort}.py"

    # Create new content with updated parameters
    new_content = original_content
    for i, param_name in enumerate(parameter_names):
        param_value = param_values[i]
        # Use regex to replace parameter values more safely
        import re
        pattern = rf'^({param_name}\s*=\s*)([^#\n]*)'
        replacement = rf'\g<1>{param_value}'
        new_content = re.sub(pattern, replacement, new_content, flags=re.MULTILINE)

    # Write the updated content to a worker-specific temporary file
    worker_param_file = f"input/parameters{cohort}_{worker_id}.py"
    with open(worker_param_file, 'w') as f:
        f.write(new_content)

    if verbose:
        original_param_dict = {}
        for i, param_name in enumerate(parameter_names):
            # Extract original value from the original content
            import re
            pattern = rf'^{param_name}\s*=\s*([^#\n]*)'
            matches = re.findall(pattern, original_content, flags=re.MULTILINE)
            original_param_dict[param_name] = matches[0].strip() if matches else "unknown"

        param_dict = dict(zip(parameter_names, param_values))

        # Create table data for parameter comparison
        headers = list(parameter_names)
        original_values = [f"{float(original_param_dict[name]):.4f}" for name in headers]
        updated_values = [f"{param_dict[name]:.4f}" for name in headers]

        table_data = [original_values, updated_values]
        row_labels = [f"Original {worker_id}", f"Updated {worker_id}"]
        # add random sleep so that different workers don't print the table at the same time
        sleep(random.uniform(0, 1))
        print(tabulate(table_data, headers=headers, showindex=row_labels, tablefmt="grid"))

    # Temporarily modify the cohort variable to point to our worker-specific file
    original_cohort = cohorts.cohort
    worker_cohort = f"{original_cohort}_{worker_id}"

    # Set cohort to worker-specific version
    cohorts.cohort = worker_cohort

    # Reload main parameters module to pick up new cohort
    if 'parameters' in sys.modules:
        importlib.reload(sys.modules['parameters'])
    else:
        __import__('parameters')

    cohorts.cohort = original_cohort

    try:
        # Run the model with worker-specific parameters but with the original cohort value
        tic = perf_counter()
        objective_value = objective_function()
        toc = perf_counter()
        if verbose:
            print(f"Worker {worker_id}: Objective value: {objective_value} (computed in {toc - tic:.2f} seconds)")
    except Exception as model_error:
        print(f"Error: Worker {worker_id}: Model evaluation failed: {model_error}")
        objective_value = 1e6
    finally:
        # Clean up worker-specific files
        try:
            os.remove(worker_param_file)
            if verbose:
                print(f"Worker {worker_id}: Cleaned up temporary worker files")
        except Exception as cleanup_error:
            print(f"Warning: Worker {worker_id}: Failed to cleanup temporary worker files: {cleanup_error}")
        return objective_value


def print_results(parameter_names):
    def print_results_callback(intermediate_result):
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Success: {intermediate_result.success}")
        print(f"Status: {intermediate_result.message}")
        print(f"Iterations: {intermediate_result.nit}")
        print(f"Function evaluations: {intermediate_result.nfev}")
        print(f"Final objective value: {intermediate_result.fun:.6f}")
        print("Optimal parameters:")
        headers = list(parameter_names)
        param_values = []
        param_values.append([f"{intermediate_result.x[i]:.6f}" for i in range(len(parameter_names))])
        print(tabulate(param_values, headers=headers, tablefmt="grid"))
        #for i, param_name in enumerate(parameter_names):
        #    optimal_val = intermediate_result.x[i]
        #    print(f"  {param_name}: {optimal_val:.6f}")
    return print_results_callback


def usage(proc):
    print("usage: " + proc +
          "\n\t-h --help: print this message" +
          "\n\t-v --verbose: print more info" +
          "\n\t-g --parameter-group: \n\t\tomega (marriage market)\n\t\tbeta (wage)\n\t\tlambda (job offer)" +
          "\n\t-c --cohort: cohort. e.g. 1970white")
    exit(0)


def main():
    short_options = "hvg:c:"
    long_options = ["help", "verbose",  "group=", "cohort="]
    group = None
    cohort = None
    verbose = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], short_options, long_options)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        usage(sys.argv[0])

    for opt, arg in opts:
        if opt in ("-h", "--Help"):
            usage(sys.argv[0])
        elif opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("-g", "--group"):
            group = arg
        elif opt in ("-c", "--cohort"):
            cohort = arg

    if cohort is None or cohort == "":
        print("'cohort' is a mandatory parameter")
        usage(sys.argv[0])
    if group is None or group == "":
        print("'group' is a mandatory parameter")
        usage(sys.argv[0])

    print(f"configuration: running on '{cohort}' cohort")
    print(f"configuration: running on '{group}' parameter group")

    names_and_bounds = []
    if group == "omega":
        names_and_bounds = [
            ["omega1", (-2, 4)],
            ["omega2", (-2, 4)],
            ["omega3", (-2, 1)],
            ["omega4_w", (0, 1)],
            ["omega5_w", (-2, 0)],
            ["omega6_w", (-3, 2)],
            ["omega7_w", (-3, 2)],
            ["omega8_w", (-5, 2)],
            ["omega9_w", (-3, 2)],
            ["omega10_w", (-3, 2)]
        ]
    elif group == "beta":
        names_and_bounds = [
                ["beta0_w", (0, 0.2)],
                ["beta11_w", (0, 0.2)],
                ["beta12_w", (0, 0.2)],
                ["beta13_w", (0, 0.2)],
                ["beta14_w", (0, 0.2)],
                ["beta15_w", (0, 0.2)],
                ["beta21_w", (-0.01, 0)],
                ["beta22_w", (-0.01, 0)],
                ["beta23_w", (-0.01, 0)],
                ["beta24_w", (-0.01, 0)],
                ["beta25_w", (-0.01, 0)],
                ["beta31_w", (8, 10)],
                ["beta32_w", (8, 10)],
                ["beta33_w", (8, 11)],
                ["beta34_w", (8, 11)],
                ["beta35_w", (8, 11)],
                ["beta0_h", (0, 0.2)],
                ["beta11_h", (0, 0.2)],
                ["beta12_h", (0, 0.2)],
                ["beta13_h", (0, 0.2)],
                ["beta14_h", (0, 0.2)],
                ["beta15_h", (0, 0.2)],
                ["beta21_h", (-0.01, 0)],
                ["beta22_h", (-0.01, 0)],
                ["beta23_h", (-0.01, 0)],
                ["beta24_h", (-0.01, 0)],
                ["beta25_h", (-0.01, 0)],
                ["beta31_h", (8, 10)],
                ["beta32_h", (8, 10)],
                ["beta33_h", (8, 11)],
                ["beta34_h", (8, 11)],
                ["beta35_h", (8, 11)]
                ]
    elif group == "lambda":
        names_and_bounds = [
                ["lambda0_w_ft", (-2, 2)],
                ["lambda1_w_ft", (0, 0.1)],
                ["lambda2_w_ft", (0, 0.1)],
                ["lambda0_h_ft", (-2, 2)],
                ["lambda1_h_ft", (0, 0.1)],
                ["lambda2_h_ft", (0, 0.1)],
                ["lambda0_w_pt", (-2, 2)],
                ["lambda1_w_pt", (0, 0.1)],
                ["lambda2_w_pt", (0, 0.1)],
                ["lambda0_h_pt", (-2, 2)],
                ["lambda1_h_pt", (0, 0.1)],
                ["lambda2_h_pt", (0, 0.1)],
                ["lambda0_w_f", (-2, 2)],
                ["lambda1_w_f", (0, 0.1)],
                ["lambda2_w_f", (0, 0.1)],
                ["lambda0_h_f", (-2, 2)],
                ["lambda1_h_f", (0, 0.1)],
                ["lambda2_h_f", (0, 0.1)]
                ]
    else:
        print(f"Unknown parameter group '{group}'")
        usage(sys.argv[0])

    parameter_names = [item[0] for item in names_and_bounds]
    bounds = [item[1] for item in names_and_bounds]

    # Read original parameter values from the parameter file
    original_param_file = f"input/parameters{cohort}.py"
    with open(original_param_file, 'r') as f:
        original_content = f.read()

    # Extract original parameter values
    original_values = []
    for param_name in parameter_names:
        import re
        pattern = rf'^{param_name}\s*=\s*([^#\n]*)'
        matches = re.findall(pattern, original_content, flags=re.MULTILINE)
        if matches:
            original_values.append(float(matches[0].strip()))
        else:
            print(f"Warning: Could not find original value for {param_name}")
            # Use midpoint of bounds as fallback
            lower, upper = bounds[len(original_values)]
            original_values.append((lower + upper) / 2)

    try:
        n_cores = max(1, int(cpu_count()*0.8))
        print(f"Will use {n_cores} CPU cores for parallel processing")

        result = differential_evolution(
                parallel_objective_wrapper,
                bounds,
                args=(cohort, parameter_names, original_content, verbose),
                x0=original_values,
                maxiter=1000,
                popsize=20,
                seed=42,
                tol=0.05, # stop when population spread < 5%
                atol=0.1, # stop when objective diff < 0.1
                disp=not verbose, # use verbose callback instad to print progress
                callback=print_results(parameter_names) if verbose else None,
                workers=n_cores,
                polish=False, # no need to refine results via local search once found
                updating='deferred'
                )

        print_results(parameter_names)(result)

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\nError during optimization: {e}")
    finally:
        # Clean up temporary parameter files
        import glob
        pattern = f"input/parameters{cohort}_*.py"
        temp_files = glob.glob(pattern)
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                print(f"Removed temporary file: {temp_file}")
            except Exception as cleanup_error:
                print(f"Warning: Failed to remove {temp_file}: {cleanup_error}")


if __name__ == "__main__":
    main()
