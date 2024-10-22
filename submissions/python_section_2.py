import pandas as pd
import numpy as np

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    toll_ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    distance_matrix = pd.DataFrame(np.inf, index=toll_ids, columns=toll_ids)
    np.fill_diagonal(distance_matrix.values, 0)
    for _, row in df.iterrows():
        start = row['id_start']
        end = row['id_end']
        dist = row['distance']
        distance_matrix.at[start, end] = dist
        distance_matrix.at[end, start] = dist
    for k in toll_ids:
        for i in toll_ids:
            for j in toll_ids:
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]
    
    return distance_matrix




def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    unrolled_data = []
    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:
                unrolled_data.append([id_start, id_end, df.at[id_start, id_end]])
    unrolled_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])
    return unrolled_df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    reference_rows = df[df['id_start'] == reference_id]
    avg_distance = reference_rows['distance'].mean()
    lower_bound = avg_distance * 0.9
    upper_bound = avg_distance * 1.1
    valid_ids = []
    for id_start in df['id_start'].unique():
        id_start_rows = df[df['id_start'] == id_start]
        avg_distance_id_start = id_start_rows['distance'].mean()
        if lower_bound <= avg_distance_id_start <= upper_bound:
            valid_ids.append({'id_start': id_start, 'average_distance': avg_distance_id_start})
    result_df = pd.DataFrame(valid_ids)
    return result_df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.
    Args:
        df (pandas.DataFrame)
    Returns:
        pandas.DataFrame
    """
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    for vehicle, coefficient in rate_coefficients.items():
        df[vehicle] = df['distance'] * coefficient
    return df



from datetime import time
def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    rows = []
    time_ranges = [
        ('Monday', time(0, 0), 'Friday', time(10, 0), 0.8),
        ('Tuesday', time(10, 0), 'Saturday', time(18, 0), 1.2),  
        ('Wednesday', time(18, 0), 'Sunday', time(23, 59, 59), 0.8),  
        ('Saturday', time(0, 0), 'Sunday', time(23, 59, 59), 0.7)  
    ]
    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']

        for start_day, start_time, end_day, end_time, discount in time_ranges:
            moto_rate = distance * 0.8 * discount
            car_rate = distance * 1.2 * discount
            rv_rate = distance * 1.5 * discount
            bus_rate = distance * 2.2 * discount
            truck_rate = distance * 3.6 * discount
            rows.append([id_start, id_end, distance, start_day, start_time, end_day, end_time, 
                         moto_rate, car_rate, rv_rate, bus_rate, truck_rate])

    new_df = pd.DataFrame(rows, columns=['id_start', 'id_end', 'distance', 'start_day', 'start_time',
                                         'end_day', 'end_time', 'moto', 'car', 'rv', 'bus', 'truck'])

    return new_df