import re
from typing import Any, Dict, List
import polyline
import pandas as pd
from geopy.distance import geodesic


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Initialize an empty result list
    result = []
    
    # Iterate through the list in chunks of size n
    for i in range(0, len(lst), n):
        # Get the current chunk and reverse it
        chunk = lst[i:i+n]
        result.extend(chunk[::-1])
    
    return result


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}
    
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    
    # Sort the dictionary by keys (lengths) and return it
    return dict(sorted(length_dict.items()))

def flatten_dict(nested_dict: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    flattened_dict = {}

    def flatten(current_dict: Dict[str, Any], parent_key: str):
        for key, value in current_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                flatten(value, new_key)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    item_key = f"{new_key}[{index}]"
                    if isinstance(item, dict):
                        flatten(item, item_key)
                    else:
                        flattened_dict[item_key] = item
            else:
                flattened_dict[new_key] = value

    flatten(nested_dict, "")
    return flattened_dict

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])  # Append a copy of the current permutation
            return
        
        seen = set()  # To keep track of used numbers at this level
        for i in range(start, len(nums)):
            if nums[i] not in seen:  # Skip duplicates
                seen.add(nums[i])
                nums[start], nums[i] = nums[i], nums[start]  # Swap to fix the current number
                backtrack(start + 1)  # Recur for the next position
                nums[start], nums[i] = nums[i], nums[start]  # Backtrack (swap back)
    
    nums.sort()  # Sort to facilitate skipping duplicates
    result = []
    backtrack(0)
    return result


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    
    # Define the regex patterns for the different date formats
    date_patterns = [
        r'\b(\d{2})-(\d{2})-(\d{4})\b',  # dd-mm-yyyy
        r'\b(\d{2})/(\d{2})/(\d{4})\b',  # mm/dd/yyyy
        r'\b(\d{4})\.(\d{2})\.(\d{2})\b'  # yyyy.mm.dd
    ]
    
    # Combine all patterns into one
    combined_pattern = '|'.join(date_patterns)
    
    # Find all matches in the text
    matches = re.findall(combined_pattern, text)
    
    # Flatten the matches and filter out empty results
    valid_dates = []
    for match in matches:
        # Each match is a tuple where only one of the patterns will have values
        valid_dates.append(next(filter(None, match)))
    
    # Reconstruct the valid date strings from the matches
    result = []
    for match in matches:
        if match[0]:  # dd-mm-yyyy
            result.append(f"{match[0]}-{match[1]}-{match[2]}")
        elif match[3]:  # mm/dd/yyyy
            result.append(f"{match[3]}/{match[4]}/{match[5]}")
        elif match[6]:  # yyyy.mm.dd
            result.append(f"{match[6]}.{match[7]}.{match[8]}")
    
    return result

# bfbhbjfbhjbhhb
def haversine_distance(coord1, coord2):
    """
    Calculate the Haversine distance between two coordinates in meters.
    """
    return geodesic(coord1, coord2).meters

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode polyline into a list of (latitude, longitude) coordinates
    coordinates = polyline.decode(polyline_str)
    
    # Create a DataFrame from the coordinates
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Calculate distances between consecutive points
    distances = [0]  # First point has no previous point, so distance is 0
    
    for i in range(1, len(coordinates)):
        coord1 = coordinates[i - 1]
        coord2 = coordinates[i]
        distance = haversine_distance(coord1, coord2)
        distances.append(distance)
    
    # Add the distances to the DataFrame
    df['distance'] = distances
    
    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    # First transpose, then reverse rows
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    # Step 2: Create the final matrix with row and column sums excluding the element itself
    final_matrix = [[0] * n for _ in range(n)]
    
    # Precompute row sums and column sums for the rotated matrix
    row_sums = [sum(rotated_matrix[i]) for i in range(n)]
    col_sums = [sum(rotated_matrix[i][j] for i in range(n)) for j in range(n)]
    
    # Replace each element with the sum of its row and column excluding itself
    for i in range(n):
        for j in range(n):
            final_matrix[i][j] = row_sums[i] + col_sums[j] - rotated_matrix[i][j]
    
    return final_matrix




def time_check(df) -> pd.Series:
    """
    Verifies the completeness of the time data by checking whether the timestamps for each unique (id, id_2) pair 
    cover a full 24-hour period and span all 7 days of the week.
    
    Args:
        df (pandas.DataFrame): The input dataframe containing 'id', 'id_2', 'startDay', 'startTime', 'endDay', 'endTime' columns.
    
    Returns:
        pd.Series: A boolean series with a multi-index (id, id_2), indicating True if the timestamps are incomplete.
    """
    
    # Helper function to generate all minute intervals for a given day
    def generate_day_intervals():
        # Generate all minutes in a day
        return pd.date_range('00:00', '23:59', freq='T').time
    
    # Expected full set of times for all 7 days of the week
    full_coverage = {day: set(generate_day_intervals()) for day in range(7)}

    # Group the DataFrame by id and id_2
    def check_group(group):
        # Dictionary to track time coverage per day of the week
        coverage = {day: set() for day in range(7)}
        
        # Iterate over each row in the group and add the intervals
        for _, row in group.iterrows():
            start_day, start_time = row['startDay'], row['startTime']
            end_day, end_time = row['endDay'], row['endTime']
            
            if start_day == end_day:  # Interval within the same day
                coverage[start_day].update(pd.date_range(start_time, end_time, freq='T').time)
            else:
                # Split the interval across multiple days
                coverage[start_day].update(pd.date_range(start_time, '23:59', freq='T').time)
                coverage[end_day].update(pd.date_range('00:00', end_time, freq='T').time)
        
        # Check if every day has full coverage
        for day in range(7):
            if coverage[day] != full_coverage[day]:
                return True  # Missing time intervals
        
        return False  # All days have full coverage
    
    # Apply the check_group function to each group and return a boolean Series
    return df.groupby(['id', 'id_2']).apply(check_group)
