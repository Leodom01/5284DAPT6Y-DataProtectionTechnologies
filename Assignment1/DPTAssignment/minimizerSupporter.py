from collections import Counter
import numpy as np


# Get the anonymity set size of the set with the same set of quasi identifiers
def get_anon_set_size(dataset, quasi_id_col_indexes):
    """
        Calculates anonymity set size metric for a given dataset.

        Anonymity set size defines how many entries with the same set of quasi-identifiers belongs to the dataset.
        This function computes the minimum, maximum, and average size of anonymity sets within
        a dataset based on the specified quasi-identifier columns.
        A set is a group of entries with the same quasi-identifier.
        A similar metric could have been the k-anonymity, but it would be hard to interpret it since a set with only one entry
        could bring the whole dataset's k-anonymity down to 1, even if the rest of the dataset has high anonymity values.

        Parameters:
        - dataset (numpy.ndarray): The input dataset containing sensitive and quasi-identifier columns. Data should
                                    be numeric.
        - quasi_id_col_indexes (list): A list integer representing indexes of the quasi-identifier columns.

        Returns:
        - min_anon_set_size (int): The minimum anonymity value found in at least one set in the dataset.
        - avg_anon_set_size (float): The average anonymity set size among all the sets in the dataset.
        - max_anon_set_size (int): The maximum anonymity size found in at least one set in the dataset.

        Example:
            dataset = np.array([
                [0, 27, 6.1],
                [0, 28, 6.2],
                [1, 29, 6.5],
                [1, 30, 7.6]
                [1, 29, 6.0],
            ])
            quasi_id_col_indexes = [0]
            min_size, avg_size, max_size = get_anon_set_size(dataset, quasi_id_col_indexes)
            print(min_size, avg_size, max_size)
            > 2 2.5 3
        2 because [0, 27, 6.1], [0, 28, 6.2] are in the set with the least number of entries.
        3 because [1, 29, 6.5], [1, 30, 7.6], [1, 29, 6.0] belong to the set with the highest number of entries.
        2.5 is the average of the set sizes.

        Note:
        - An anonymity set refers to a group of records that have the same values in the quasi-identifier
          columns. The size of an anonymity set is the number of records in that group.

        """
    counter = Counter()

    # Count all entries with the same quasi_identifier (count the set size)
    for entry in dataset[:, quasi_id_col_indexes]:
        counter[tuple(entry)] += 1

    # Get the set with the smallest and biggest size
    # It could be interesting to have a weighted average that takes into account the number of entries in a dataset
    min_anon_set_size = counter.most_common()[-1][1]
    max_anon_set_size = counter.most_common(1)[0][1]
    avg_anon_set_size = np.mean(list(counter.values()))

    return min_anon_set_size, avg_anon_set_size, max_anon_set_size,


def add_fake_entries(dataset, new_entries_percentage, quasi_identifiers_idx, sensitive_data_col_idx):
    """
        Adds synthetic entries to a dataset while preserving the major statistical properties.

        This function adds synthetic entries to an existing dataset while maintaining
        statistical characteristics of the original data. It allows for preserving the distribution
        of quasi-identifiers while adding diversity to sensitive data columns.
        The goal is to increase the average anonymity set size (and eventually the k-anonymity) and enhance the
        l-diversity.
        The anonymity set size is increased by generating entries with the same set of quasi-identifiers of
        original record.
        The l-diversity is achieved or at least improved by generating new sensitive values slightly different from
        the ones already available in records with the same quasi-identifiers.
        Non-sensitive and non-quasi-identifiers features (if available) will be added following a normal distribution
        with mean and standard deviation equal to the real data.

        Parameters:
        - dataset (numpy.ndarray): The original dataset to which synthetic entries will be added.
        - new_entries_percentage (float): The percentage of new synthetic entries relative to the original dataset.
        - quasi_identifiers_idx (list): A list of column indices representing quasi-identifier columns.
        - sensitive_data_col_idx (list): A list of column indices representing sensitive data columns.

        Returns:
        - enriched_dataset (numpy.ndarray): The dataset with synthetic entries added.

        Example:
            original_dataset = np.array([
                [0, 27, 6.1, 126],
                [0, 28, 6.2, 125],
                [1, 29, 6.5, 132],
                [1, 30, 7.6, 140]
            ])
            quasi_idx = [0, 1]
            sensitive_idx = [3]
            synthetic_data = add_fake_entries(original_dataset, 0.2, quasi_idx, sensitive_idx)
        Returns an array with one added value with quasi-identifier picked randomly among:
        [0, 27], [0, 28], [1, 29], [1, 30]

        """

    # Calculate the number of new entries to add
    num_new_entries = int(dataset.shape[0] * new_entries_percentage)

    # Create an array to store new synthetic entries
    synthetic_entries = []

    # A list containing the index of a dataset entry picked randomly. The i-th generated entry will inherit the
    # quasi-identifier from the entry in the dataset with index: entry_quasi_id_map[i]
    entry_quasi_id_map = [np.random.randint(0, dataset.shape[0]) for _ in range(num_new_entries)]

    for col_index in range(dataset.shape[1]):

        # Calculate stats for every feature to maintain the major statistical properties.
        col_mean = np.mean(dataset[:, col_index])
        col_std = np.std(dataset[:, col_index])
        col_max = np.max(dataset[:, col_index])
        col_min = np.min(dataset[:, col_index])

        # Assign to every new entry the quasi_identifier values of the dataset entry with the same index as the
        # one in entry_quasi_id_map
        if col_index in quasi_identifiers_idx:
            new_column = np.array([dataset[dataset_index, col_index] for dataset_index in entry_quasi_id_map])

        # Every new sensitive data has to be similar to the sensitive data of the entry with the same quasi-identifiers
        elif col_index in sensitive_data_col_idx:
            new_column = np.array(
                [dataset[dataset_index, col_index] + np.random.normal(dataset[dataset_index, col_index], col_std)
                 for dataset_index in entry_quasi_id_map])

        # If a feature is not sensitive nor quasi-identifier assign to it a random value following
        # mean and std of that feature
        else:
            new_column = np.random.normal(col_mean, col_std, num_new_entries)

            # All new values have to fall within the old values range.
            # This prevents an attacker to notice clear differences (negative values or values above the max value),
            # since we don't know the semantic of the feature we want to play it safe and stay between the tracks.
            for i in range(len(new_column)):
                if new_column[i] < col_min or new_column[i] > col_max:
                    new_column[i] = np.random.uniform(col_min, col_max)

        synthetic_entries.append(new_column)

    # Convert the synthetic entries to a NumPy array
    synthetic_data = np.array(synthetic_entries).T

    # Concatenate the synthetic data with the original dataset
    enriched_dataset = np.concatenate((dataset, synthetic_data))

    return enriched_dataset

def get_l_diversity(dataset, quasi_identifiers_idx):
    """
        Calculate L-diversity metrics for a given dataset.

        L-diversity measures the diversity of sensitive values in a dataset relative to quasi-identifier values.
        The logic of the function is to calculate the number of distinct sensitive values in every single set (a set is
        a group of entries with the same quasi-identifiers).
        Every set will have an l-diversity percentage, 100% means that the set is l-diverse, the percentage is obtained
        with the following formula: 100*<number of distinct sensitive values>/<number of entries in the set>.
        Knowing whether a dataset is l-diverse according to the standard definition is useful but cannot provide
        a gradient. The standard l-diversity definition tells us if a dataset is l-diverse or not, we also want to know
        the possible shades: min, avg e max allow us to know that.
        The function returns the minimum, mean, and maximum L-diversity percentages.

        Parameters:
            dataset (numpy.ndarray): The dataset containing both quasi-identifier and sensitive information.
            quasi_identifiers_idx (list): A list of column indices representing quasi-identifiers, every other feature
                                        is considered sensitive.

        Returns:
            tuple: A tuple containing the minimum, mean, and maximum L-diversity percentages.

        Example:
            dataset = np.array([
                [30, 'A', 'X'],
                [30, 'A', 'Y'],
                [30, 'A', 'Z'],
                [25, 'B', 'Y'],
                [25, 'B', 'X'],
                [25, 'B', 'X'],
            ])
            quasi_identifiers_idx = [0, 1]
            min_l_div, avg_l_div, max_l_div = get_l_diversity(dataset, quasi_identifiers_idx)
            print(min_l_div, avg_l_div, max_l_div)
            > 67, 84, 100
        The entries with 30 and A as quasi-identifiers: [30, 'A', 'X'], [30, 'A', 'Y'], [30, 'A', 'Z'] have 3 distinct
        sensitive values, therefore the l-diversity is 3/3 = 1  (100% -> max l-div)
        The entries with 25 and B as quasi-identifiers: [25, 'B', 'Y'], [25, 'B', 'X'] [25, 'B', 'X'] have two distinct
        sensitive values, therefore the l-diversity is 2/3 = 0.67  (67% -> min l-div)
        The mean among the two only set available is 0.835  (84% -> average l-div)
        """

    # Dictionary to group the entries by quasi-identifiers. The key will be the quasi-identifiers set and the
    # value will be a list of sensitive values sets
    vals_in_set = dict()

    for entry in dataset:
        # Get the dictionary key based on quasi_identifiers
        key = np.array2string(entry[quasi_identifiers_idx])
        # Get the sensitive values set to add
        to_add = np.array2string(entry[[i for i in range(len(entry)) if i not in quasi_identifiers_idx]])

        # If there's already a value for the key then add the current sensitive values set
        if key in vals_in_set.keys():
            vals_in_set[key].append(to_add)
        # If the current key has no dictionary value then create a list with the first sensitive values set
        else:
            vals_in_set[key] = [to_add]

    # For each set (group of entries with the same quasi-identifiers) calculate the l-div percentage
    # L-div percentage is calculated with: 100*<number of distinct sensitive values>/<number of records in the set>
    l_div_percentage = []

    # Fill the l_div_percentage list according to the previously mentioned formula
    for key in vals_in_set.keys():
        l_div_percentage.append(
            # len(set(vals_in_set[key])) thanks to the set() returns the number of distinct sensitive values
            100 * float(len(set(vals_in_set[key]))) / float(len(vals_in_set[key]))
        )

    # Convert to numpy array to quickly calculate min, avg and max
    l_div_percentage = np.array(l_div_percentage)

    return np.min(l_div_percentage), np.mean(l_div_percentage), np.max(l_div_percentage)
