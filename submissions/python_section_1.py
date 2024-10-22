from typing import Dict, List
import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    i = 0
    while i < len(lst):
        group = []
        for j in range(i, min(i + n, len(lst))):
            group.append(lst[j])
        for j in range(len(group) - 1, -1, -1):
            result.append(group[j])
        i += n
    return result


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    result = {}
    for s in lst:
        length = len(s)
        if length not in result:
            result[length] = []
        result[length].append(s)
    return dict(sorted(result.items()))



def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    if not nested_dict:
        return {}
    flattened = {}
    stack = [(nested_dict, '')]
    while stack:
        current_dict, parent_key = stack.pop()
        for k, v in current_dict.items():
            new_key = f'{parent_key}{sep}{k}' if parent_key else k
            if isinstance(v, dict):
                stack.append((v, new_key))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        stack.append((item, f'{new_key}[{i}]'))
                    else:
                        flattened[f'{new_key}[{i}]'] = item
            else:
                flattened[new_key] = v
    return flattened



def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start):
        if start == len(nums):
            results.append(nums[:])
            return
    
        seen = set()
        for i in range(start, len(nums)):
            if nums[i] in seen:
                continue
            seen.add(nums[i])
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]
    nums.sort()
    results = []
    backtrack(0)
    return results


import re
def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    pattern = r'(\b\d{2}-\d{2}-\d{4}\b|\b\d{2}/\d{2}/\d{4}\b|\b\d{4}\.\d{2}\.\d{2}\b)'
    matches = re.findall(pattern, text)
    return matches



import polyline
import numpy as np
def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    
    def haversine(lat1, lon1, lat2, lon2):
        """
        Calculate the great-circle distance between two points on the Earth specified in decimal degrees
        using the Haversine formula.
        
        Args:
            lat1 (float): Latitude of point 1.
            lon1 (float): Longitude of point 1.
            lat2 (float): Latitude of point 2.
            lon2 (float): Longitude of point 2.
        
        Returns:
            float: Distance in meters between the two points.
        """
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a)) 
        
        r = 6371000  
        return c * r

    if not polyline_str:
        raise ValueError("The polyline string cannot be empty.")
    
    try:
        coordinates = polyline.decode(polyline_str)
    except Exception as e:
        raise ValueError(f"Error decoding polyline string: {e}")
    
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    distances = [0] 
    for i in range(1, len(df)):
        lat1, lon1 = df.iloc[i - 1]
        lat2, lon2 = df.iloc[i]
        distance = haversine(lat1, lon1, lat2, lon2)
        distances.append(distance)
    
    df['distance'] = distances
    pd.options.display.float_format = '{:,.6f}'.format
    return df



def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - i - 1] = matrix[i][j]

    final_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(row[j] for row in rotated_matrix) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum

    return final_matrix




def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period
    Args:
        df (pandas.DataFrame)
    Returns:
        A boolean series indicating if each (id, id_2) pair has correct timestamps (True for correct, False for incorrect).
    """

    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], format='%A %H:%M:%S')
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], format='%A %H:%M:%S')

    grouped = df.groupby(['id', 'id_2'])
    def check_group(group):
        days_covered = group['start_timestamp'].dt.day_name().unique()
        
        all_days = set(pd.Series(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']))
        complete_days = all_days.issubset(days_covered)

        min_time = group['start_timestamp'].min()
        max_time = group['end_timestamp'].max()

        full_24_hours = (max_time - min_time) >= pd.Timedelta(days=1)

        return complete_days and full_24_hours

    result = grouped.apply(check_group)
    return result.rename_axis(['id', 'id_2']).astype(bool)
