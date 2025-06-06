import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import stats


# Display settings for debugging large DataFrames
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def load_and_clean_data(filepath: str) -> dict:
    """
    Loads and preprocesses the data from CSV.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        dict: Cleaned and parsed data dictionary, keyed by participant ID.
    """
    data = pd.read_csv(filepath, sep=',')

    # Remove incomplete entries
    data['endhit'].replace('', np.nan, inplace=True)
    data['hitid'].replace('HIT_ID', np.nan, inplace=True)
    data.dropna(subset=['endhit'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Filter for condition 0: participants who saw examples
    original_data = data[data["cond"] == 0]

    # Convert JSON strings to dictionaries
    parsed_data = {}
    datastrings = original_data["datastring"].to_dict()
    for key, value in datastrings.items():
        parsed_data[key] = json.loads(value)

    return parsed_data


def extract_correct_answers_and_bonus(data: dict) -> pd.DataFrame:
    """
    Extracts number of correct answers and corresponding bonuses.

    Args:
        data (dict): Parsed participant data.

    Returns:
        pd.DataFrame: DataFrame containing correct answer counts and bonus.
    """
    correct_answers = []
    bonuses = []

    for pid, record in data.items():
        trial_index = {
            49: 14,
            51: 16
        }.get(record.get("currenttrial"), 18)

        try:
            correct = sum(record['data'][trial_index]['trialdata']['correct'])
        except (KeyError, IndexError, TypeError):
            correct = 0

        try:
            bonus = record['questiondata']['final_bonus']
        except (KeyError, TypeError):
            bonus = 0

        correct_answers.append(correct)
        bonuses.append(bonus)

    return pd.DataFrame({
        "Correct answers": correct_answers,
        "Bonus": bonuses
    })


def analyze_and_plot(df: pd.DataFrame):
    """
    Computes and prints correlation stats, and plots results.

    Args:
        df (pd.DataFrame): DataFrame with correct answer counts and bonuses.
    """
    print(df)

    stat, p = stats.spearmanr(df["Correct answers"], df["Bonus"])
    print(f"Spearman correlation: {stat:.3f}, p-value: {p:.4f}")

    plt.plot(df["Correct answers"], df["Bonus"], 'ro')
    plt.xlabel('Correct answers')
    plt.ylabel('Bonus')
    plt.title('Bonus vs Correct Answers')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    filepath = "dataclips.csv"
    participant_data = load_and_clean_data(filepath)
    df_scores = extract_correct_answers_and_bonus(participant_data)
    analyze_and_plot(df_scores)
