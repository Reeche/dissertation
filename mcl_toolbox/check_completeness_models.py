import os
import itertools
import pandas as pd

"""
This script checks whether all model fits (per PID and model index) exist 
for a given experiment and model type. It outputs missing combinations 
as a CSV file for further inspection.
"""

# --------------------- Data Dictionaries --------------------- #

# Mapping of PIDs per model type and experiment
PID_DICTS = {
    'hybrid': {
        'strategy_discovery': [3, 4, 6, 7, 9, 16, 17, 19, 23, 30, 34, 35, 41, 45, 53, 57, 58, 67, 71, 76,
                               78, 83, 86, 92, 106, 128, 133, 138, 139, 141, 143, 146, 155, 161, 164, 165,
                               167, 174, 175, 177, 184, 189, 194, 195, 201, 203, 206, 211, 216, 218, 219, 223,
                               228, 231, 232, 236, 238, 250, 255, 259, 260, 262, 267, 280, 281, 291, 292, 293,
                               299, 305, 310, 316, 317, 318, 320, 324, 327, 328, 341, 344, 347, 349, 350, 355,
                               356, 357, 359, 360, 361, 362, 373, 374, 375, 377],
    },
    'mf': {
        'strategy_discovery': [2, 8, 24, 43, 48, 49, 54, 62, 68, 73, 75, 80, 85, 91, 93, 96, 99, 102, 107, 110, 113, 116,
                               117, 120, 123, 124, 126, 131, 137, 145, 147, 149, 153, 156, 159, 166, 169, 171, 172, 178,
                               181, 183, 185, 187, 190, 199, 200, 207, 212, 213, 220, 221, 226, 229, 233, 242, 244, 246,
                               247, 252, 261, 263, 266, 274, 279, 286, 287, 294, 295, 296, 306, 319, 333, 337, 340, 365,
                               367, 369, 372, 376, 378]
    },
    'habitual': {
        'strategy_discovery': [1, 10, 11, 14, 20, 22, 25, 26, 27, 29, 33, 36, 37, 39, 40, 46, 50, 51, 52, 55, 59, 65, 70,
                               89, 95, 98, 101, 111, 115, 118, 119, 125, 129, 134, 135, 140, 142, 148, 151, 154, 162, 170,
                               180, 186, 192, 193, 202, 204, 205, 209, 210, 214, 215, 217, 234, 235, 237, 240, 241, 249,
                               253, 254, 257, 265, 268, 271, 276, 277, 282, 289, 300, 304, 308, 312, 313, 321, 322, 323,
                               329, 330, 331, 332, 339, 343, 348, 358, 363, 364, 370]
    },
    'non_learning': {
        'strategy_discovery': [18, 28, 32, 38, 56, 63, 72, 77, 82, 90, 103, 109, 122, 152, 173, 196, 239, 256, 275, 278,
                               309, 311, 315, 335, 336, 342, 346, 352, 353, 354, 371]
    }
}

# Model indices associated with each model type
MODEL_INDICES = {
    'habitual': [1743],
    'mf': [491],
    'hybrid': [3326],
    'non_learning': [1756]
}

# --------------------- Utility Functions --------------------- #

def get_pid_list(model_type: str, exp: str) -> list:
    """Retrieve the list of participant IDs for a given model type and experiment."""
    return PID_DICTS[model_type][exp]

def get_filename_combinations(pids: list, model_indices: list) -> set:
    """Create all expected filename combinations (pid, model_index)."""
    return {f"{pid},{mid}" for pid, mid in itertools.product(pids, model_indices)}

def extract_existing_files(directory: str) -> set:
    """Extract PID and model_index pairs from filenames in the given directory."""
    filenames = os.listdir(directory)
    return {
        f.replace("likelihood_", "").replace(".pkl", "").replace("_", ",")
        for f in filenames
    }

def find_missing_items(expected: set, existing: set) -> list:
    """Return list of missing (pid, model_index) combinations."""
    missing = sorted(expected - existing)
    return [tuple(map(int, item.split(","))) for item in missing]

def save_missing_items(model_type: str, missing: list):
    """Save the list of missing fits to a CSV file."""
    df = pd.DataFrame(missing, columns=['pid', 'model_index'])
    df.to_csv(f"missing_{model_type}.csv", index=False)

def check_model_fits(exp_name: str, model_types: list):
    """Main function to check for missing model fits across model types."""
    for model_type in model_types:
        print(f"\nChecking model type: {model_type}")
        pids = get_pid_list(model_type, exp_name)
        model_indices = MODEL_INDICES[model_type]
        expected_combinations = get_filename_combinations(pids, model_indices)

        results_dir = f"results_model_recovery_sd/{model_type}/{exp_name}_priors"
        existing_combinations = extract_existing_files(results_dir)

        missing_items = find_missing_items(expected_combinations, existing_combinations)

        print(f"Number of missing items for {model_type}: {len(missing_items)}")
        print(f"Missing items for {exp_name}:", missing_items)

        save_missing_items(model_type, missing_items)

# --------------------- Run Check --------------------- #

if __name__ == "__main__":
    experiment_name = "strategy_discovery"
    models_to_check = ["non_learning", "habitual", "mf", "hybrid"]
    check_model_fits(experiment_name, models_to_check)
