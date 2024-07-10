# print(data)
# print(data.info())
# print(data.describe)
# print(data.shape)


# # Adjust display options to show all rows and columns
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# # Assuming 'df' is your DataFrame
# summary = data.describe()

# print(summary)


# # Define the numerical columns
# numerical_columns = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']

# # Plot histograms
# plt.figure(figsize=(15, 10))
# for i, column in enumerate(numerical_columns, 1):
#     plt.subplot(2, 3, i)
#     plt.hist(data[column].dropna(), bins=30, edgecolor='black')
#     plt.title(column)
#     plt.xlabel(column)
#     plt.ylabel('Frequency')
# plt.tight_layout()
# plt.show()

# categorical_columns = ['neighbourhood_group', 'neighbourhood', 'room_type', 'last_review']

# for column in categorical_columns:
#     print(f"Unique values in {column}: {data[column].nunique()}")


# Missing values

# missing_values = data.isnull().sum()
# print(missing_values)


# plt.figure(figsize=(10, 6))
# sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
# plt.title('Heatmap of Missing Values')
# plt.show()

# # Identify Outliers
# numerical_columns = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']

# # Box plots to identify outliers
# plt.figure(figsize=(15, 10))
# for i, column in enumerate(numerical_columns, 1):
#     plt.subplot(2, 3, i)
#     sns.boxplot(data[column])
#     plt.title(column)
# plt.tight_layout()
# plt.show()

# # Function to detect outliers using IQR
# def detect_outliers(data, column):
#     Q1 = data[column].quantile(0.25)
#     Q3 = data[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
#     return outliers

# # Print the number of outliers in each numerical column
# for column in numerical_columns:
#     outliers = detect_outliers(data, column)
#     print(f'Number of outliers in {column}: {len(outliers)}')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Step 1: Load the dataset
data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")

# Handling Missing Values
data['reviews_per_month'] = data['reviews_per_month'].fillna(data['reviews_per_month'].median())
data['name'] = data['name'].fillna('Unknown')
data['host_name'] = data['host_name'].fillna('Unknown')

# Handle missing 'last_review' by converting to a numeric format
data['last_review'] = pd.to_datetime(data['last_review'])
most_recent_date = data['last_review'].max()
data['last_review'] = data['last_review'].fillna(most_recent_date)
data['days_since_last_review'] = (most_recent_date - data['last_review']).dt.days
data.drop('last_review', axis=1, inplace=True)

# Separate numeric and non-numeric columns
numeric_columns = data.select_dtypes(include=[np.number]).columns
non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns

print("Numeric Columns:\n", numeric_columns)
print("Non-numeric Columns:\n", non_numeric_columns)

# Normalize numeric columns using MinMaxScaler
scaler = MinMaxScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

print("\nData after normalization:\n", data[numeric_columns].head())

# Encode non-numeric columns (one-hot encoding)
data = pd.get_dummies(data, columns=non_numeric_columns, drop_first=True)

print("\nData after one-hot encoding:\n", data.head())

# Treating Outliers in 'price'
Q1 = data['price'].quantile(0.25)
Q3 = data['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Print IQR bounds for 'price'
print("\nPrice IQR Lower Bound:", lower_bound)
print("Price IQR Upper Bound:", upper_bound)

data['price'] = data['price'].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))

print("\nData after treating outliers in 'price':\n", data[['price']].head())

# Optional: For better handling of outliers, consider using RobustScaler instead of MinMaxScaler
robust_scaler = RobustScaler()
data[numeric_columns] = robust_scaler.fit_transform(data[numeric_columns])

print("\nData after robust scaling:\n", data[numeric_columns].head())


# -------------------------------------------------------------------------
#  #   Column                          Non-Null Count  Dtype
# ---  ------                          --------------  -----
#  0   id                              48895 non-null  int64
#  1   name                            48879 non-null  object
#  2   host_id                         48895 non-null  int64
#  3   host_name                       48874 non-null  object
#  4   neighbourhood_group             48895 non-null  object
#  5   neighbourhood                   48895 non-null  object
#  6   latitude                        48895 non-null  float64
#  7   longitude                       48895 non-null  float64
#  8   room_type                       48895 non-null  object
#  9   price                           48895 non-null  int64
#  10  minimum_nights                  48895 non-null  int64
#  11  number_of_reviews               48895 non-null  int64
#  12  last_review                     38843 non-null  object
#  13  reviews_per_month               38843 non-null  float64
#  14  calculated_host_listings_count  48895 non-null  int64
#  15  availability_365                48895 non-null  int64

# Numerical Features

# These are features that contain numerical values:

#     id: Unique identifier for the listing.
#     host_id: Unique identifier for the host.
#     latitude: Latitude coordinate of the listing.
#     longitude: Longitude coordinate of the listing.
#     price: Price per night for the listing.
#     minimum_nights: Minimum number of nights required to book the listing.
#     number_of_reviews: Total number of reviews for the listing.
#     reviews_per_month: Average number of reviews per month.
#     calculated_host_listings_count: Total number of listings the host has.
#     availability_365: Number of days in a year the listing is available for booking.

# Categorical Features

# These are features that contain categorical values:

#     name: Name of the listing.
#     host_name: Name of the host.
#     neighbourhood_group: General area or district.
#     neighbourhood: Specific neighborhood.
#     room_type: Type of room offered (e.g., entire home/apt, private room, shared room).
#     last_review: Date of the most recent review (although it's a date, it can be treated categorically in some contexts).

# missing_values

# id                                    0
# name                                 16
# host_id                               0
# host_name                            21
# neighbourhood_group                   0
# neighbourhood                         0
# latitude                              0
# longitude                             0
# room_type                             0
# price                                 0
# minimum_nights                        0
# number_of_reviews                     0
# last_review                       10052
# reviews_per_month                 10052
# calculated_host_listings_count        0
# availability_365                      0