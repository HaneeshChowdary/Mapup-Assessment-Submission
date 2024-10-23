import pandas as pd
import numpy as np

def calculate_distance_matrix(filepath: str) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataset from the given file path.

    Args:
        filepath (str): The path to the CSV file containing the dataset.

    Returns:
        pandas.DataFrame: Distance matrix
    """

    # Load the dataset
    df = pd.read_csv(filepath)

    # Extract unique toll locations
    locations = sorted(set(df['id_start']).union(df['id_end']))

    # Initialize the distance matrix with np.inf for unknown distances and 0 for diagonal elements
    distance_matrix = pd.DataFrame(np.inf, index=locations, columns=locations)
    np.fill_diagonal(distance_matrix.values, 0)

    # Populate the distance matrix with known distances
    for _, row in df.iterrows():
        start, end, dist = row['id_start'], row['id_end'], row['distance']
        distance_matrix.at[start, end] = dist
        distance_matrix.at[end, start] = dist  # Ensuring symmetry

    # Use Floyd-Warshall algorithm to calculate the shortest distances (cumulative distances)
    for k in locations:
        for i in locations:
            for j in locations:
                distance_matrix.at[i, j] = min(distance_matrix.at[i, j], distance_matrix.at[i, k] + distance_matrix.at[k, j])

    return distance_matrix



def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Initialize an empty list to store rows of the unrolled data
    unrolled_data = []

    # Iterate over the DataFrame to unroll it
    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:  # Exclude same start and end points (diagonal)
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': df.at[id_start, id_end]})

    # Convert the unrolled data into a new DataFrame
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df




def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Step 1: Calculate the average distance for the reference_id
    ref_distances = df[df['id_start'] == reference_id]['distance']
    ref_avg_distance = ref_distances.mean()

    # Step 2: Calculate the 10% threshold (floor and ceiling)
    lower_threshold = ref_avg_distance * 0.9
    upper_threshold = ref_avg_distance * 1.1

    # Step 3: Calculate the average distance for all IDs and filter based on the threshold
    id_avg_distances = df.groupby('id_start')['distance'].mean()
    
    # Step 4: Find IDs whose average distance falls within the 10% range
    within_threshold_ids = id_avg_distances[
        (id_avg_distances >= lower_threshold) & (id_avg_distances <= upper_threshold)
    ].index.tolist()

    # Step 5: Sort and return the IDs
    return sorted(within_threshold_ids)



def calculate_toll_rate(df):
    # Toll rate coefficients for each vehicle type
    coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate toll rates for each vehicle type based on the distance
    df['moto'] = df['distance'] * coefficients['moto']
    df['car'] = df['distance'] * coefficients['car']
    df['rv'] = df['distance'] * coefficients['rv']
    df['bus'] = df['distance'] * coefficients['bus']
    df['truck'] = df['distance'] * coefficients['truck']
    
    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

    return df
