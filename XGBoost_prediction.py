# Import statements
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# geospatial analysis files
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points

# Data Processing Imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Pipeline Imports
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer

# Model Imports
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# Assessment Metrics
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance

# saving models
import pickle
import joblib

'''
Data Pre-processing
'''
print("Importing Files")

# Path to Files
# listing for LA AirBnB homes (missing distance to coast)
listing = pd.read_csv('la_listing.csv')
# get the first 5 rows of the dataset
print(listing.head())

print("Data Cleaning")
# Step 1: Convert categorical type to string
listing['price'] = listing['price'].astype(str)
# Step 2: Clean the price columns (remove dollar sign and commas)
listing['price'] = listing['price'].replace({'\$': '', ',': ''}, regex=True)
# Step 3: Convert to numeric, coercing any errors to NaN
listing['price'] = pd.to_numeric(listing['price'], errors='coerce')

# Dropping Unncessary Features
drop_features = ['listing_url', 'scrape_id', 'last_scraped', 'source', 'picture_url', 'host_id', 'host_url', 'host_name', 'host_location', 'host_about', 'host_thumbnail_url', 'host_picture_url', 'host_has_profile_pic', 'bathrooms', 'has_availability', 'availability_30', 'availability_60', 'availability_90', 'availability_365', 'calendar_last_scraped', 'calendar_updated', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'first_review', 'last_review', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'license', 'reviews_per_month']

# drop features
listing = listing.drop(columns=drop_features)

# get the cont columns
cont = listing.select_dtypes(include=['int64', 'float64']).columns

# remove any highly correlated features
corr = listing[cont].corr()
# setting 0.7 as the threshold
threshold = 0.7
# select feature names with correlation of 70% or higher with 'price'
high_corr_features = corr.loc[abs(corr['price']) >= threshold, 'price'].index.tolist()

print("Highly Correlated Features: ")
print(high_corr_features)

print(listing.head())

'''
Exploratory Data Analysis
'''
## LISTING DATASET
# General Overview
print("Dataset Shape:", listing.shape)
print("\nColumn Information:")
print(listing.info())

# Summary Statistics for Numerical Columns
print("\nSummary Statistics:")
print(listing.describe())

# Checking Missing Values
print("\nMissing Values:")
print(listing.isnull().sum())

# Checking Unique Values for Categorical Columns
categorical_cols = listing.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    print(f"\nUnique values in '{col}':")
    print(listing[col].value_counts())

# Get a list of continuous features (based on dtype 'int64' or 'float64')
cont = listing.select_dtypes(include=['int64', 'float64']).columns.tolist()

cont_df = listing[cont]

def show_heatmap(data, output_filename):
    """
    Create a correlation heatmap for the given dataset.
    
    Args:
        data (pd.DataFrame): Input DataFrame to compute correlations
        output_filename (str): Path where the heatmap will be saved
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Compute the correlation matrix
    corr_matrix = data.corr()

    # Create the figure and axes with increased size to accommodate annotations
    plt.figure(figsize=(25, 15))  # Increased figure size

    # Use seaborn for a more aesthetic heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True,  # Add correlation values in each cell
        cmap='coolwarm',  # Diverging colormap (blue to red)
        center=0,  # Center colormap at 0
        square=True,  # Make the plot square
        linewidths=0.5,  # Add lines between cells
        cbar_kws={"shrink": .8},  # Slightly reduce colorbar size
        fmt=".2f",  # Format correlation values to 2 decimal places
        annot_kws={
            "size": 8,  # Adjust font size
            "weight": "bold",  # Make text bold
            "ha": "center",  # Horizontal alignment
            "va": "center"  # Vertical alignment
        }
    )

    plt.title("Feature Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    
    # Close the plot to free up memory
    plt.close()

show_heatmap(cont_df, "XGBoost_heatmap.png")

def show_box_whisker_plots(data, output_filename):
    """
    Create box and whisker plots for all continuous features.
    
    Args:
        data (pd.DataFrame): Input DataFrame with continuous features
        output_filename (str): Path where the plots will be saved
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Determine number of continuous features
    cont_features = data.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns

    # Calculate number of rows and columns for subplots
    n_features = len(cont_features)
    n_cols = 4  # 4 plots per row
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division

    # Create figure with subplots
    fig, axes = plt.subplots(
        nrows=n_rows, 
        ncols=n_cols, 
        figsize=(20, 4 * n_rows),  # Adjust figure size based on number of rows
        squeeze=False  # Ensure axes is always a 2D array
    )

    # Flatten axes for easier iteration
    axes = axes.ravel()

    # Create box plots for each continuous feature
    for i, feature in enumerate(cont_features):
        sns.boxplot(
            x=data[feature], 
            ax=axes[i],
            color='skyblue',
            width=0.5
        )
        
        # Set title and labels
        axes[i].set_title(f'Box Plot - {feature}')
        axes[i].set_xlabel('Value')

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    
    # Close the plot to free up memory
    plt.close()

show_box_whisker_plots(cont_df, "XGBoost_box_whiskers.png")

# Drop NaNs in crucial columns
listing = listing.dropna()

print(listing.head())

# export the final dataset
listing.to_csv("FINAL_listing_wo_dist.csv")

'''
Feature Engineering for Listing
'''
# Load the shapefile containing the coastlines
coastline = gpd.read_file('cb_2018_us_nation_20m.shp')

# Ensure the shapefile is in the correct projection (WGS 84 - EPSG:4326)
coastline = coastline.to_crs(epsg=4326)

def calculate_distance_to_coast(lat, lon, coastline):
    """
    Calculate the shortest distance to the coast for a given latitude and longitude.

    Parameters:
    - lat (float): Latitude of the point.
    - lon (float): Longitude of the point.
    - coastline (GeoDataFrame): Geospatial dataframe of the coastline.

    Returns:
    - float: Distance in meters (assuming geodetic projection).
    """
    # Create a Point object for the given latitude and longitude
    point = Point(lon, lat)

    # Find the nearest point on the coastline
    nearest_geom = coastline.geometry.union_all()  # Combine all geometries into one
    nearest_point = nearest_points(point, nearest_geom)[1]

    # Calculate distance
    distance = point.distance(nearest_point)
    return distance

# Print the unique geometry types present in the coastline GeoDataFrame
print(coastline.geom_type.unique())

# Correct any invalid geometries in the 'geometry' column by buffering with a distance of 0
coastline['geometry'] = coastline['geometry'].buffer(0)

# Replace the 'geometry' column with the boundaries of the geometries
coastline['geometry'] = coastline.boundary

# Calculate the distance from each listing to the coastline and add it as a new column 'Distance_to_Coast'
# The lambda function takes each row of the 'listing' DataFrame and uses a custom function 'calculate_distance_to_coast'
# It uses the latitude and longitude from the row and the coastline data
listing.loc[:, 'Distance_to_Coast'] = listing.apply(
    lambda row: calculate_distance_to_coast(row['latitude'], row['longitude'], coastline), axis=1
)

# Check the dataset values
print(listing.head())

# export the final dataset
listing.to_csv("FINAL_listing_w_dis.csv")

'''
CHECKPOINT
'''
listing = pd.read_csv('FINAL_listing_w_dist.csv')

print(listing.head())
print(listing.shape)

# Get data types of all columns
print(f"Listing Data Types: {listing.dtypes}")

# Get unique values
unique_values = listing['host_response_rate'].unique()
unique_values_times = listing['host_response_time'].unique()

print(f"Unique Values in Response Rate: {unique_values}")
print(f"Unique Values in Response Time: {unique_values_times}")

print(f"RESPONSE RATE: {listing['host_response_time'].value_counts()}")
print(f"SUPER HOST: {listing['host_is_superhost'].value_counts()}")

plt.figure(figsize=(8, 6))
sns.countplot(x='host_response_time', data=listing, palette='viridis')
plt.title('Distribution of Host Response Time')
plt.xlabel('Is Superhost')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate labels if necessary
plt.tight_layout()  # Adjust layout to avoid overlapping
plt.savefig("host_response_time_distribution.png")  # Save as PNG
plt.show()  # Show the plot

def handle_outliers(data, method='iqr'):
    """
    Handle outliers in the dataset using different methods.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        method (str): Method to handle outliers
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    import numpy as np

    # Create a copy of the data to avoid modifying the original
    processed_data = data.copy()

    # Determine continuous features
    cont_features = data.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns

    cont_features.drop(['longitude', 'latitude'])

    for feature in cont_features:
        # Interquartile Range (IQR) Method
        if method == 'iqr':
            Q1 = data[feature].quantile(0.25)
            Q3 = data[feature].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Remove or cap outliers
            processed_data[feature] = np.clip(data[feature], lower_bound, upper_bound)

        
        # Z-Score Method
        elif method == 'zscore':
            mean = data[feature].mean()
            std = data[feature].std()
            z_threshold = 3  # Standard deviation threshold
            
            processed_data[feature] = np.where(
                np.abs((data[feature] - mean) / std) > z_threshold,
                mean,  # Replace with mean
                data[feature]
            )
        
        # Percentile Method
        elif method == 'percentile':
            lower = data[feature].quantile(0.01)
            upper = data[feature].quantile(0.99)
            
            processed_data[feature] = data[feature].clip(lower, upper)

    return processed_data

# Choose and apply a method
listing = handle_outliers(listing, method='iqr')

print(f"HEAD: {listing.head()}")

print(f"SHAPE: {listing.shape}")
'''
XGBoost Model for Demographics Data
'''
print("XGBoost")
# Prepare X and y for demographics data
predictors = listing.columns.drop(['id', 'host_response_time']).tolist()

# Get a list of categorical features (based on dtype 'object' or low unique values)
cat = listing[predictors].select_dtypes(include=['object', 'bool']).columns.tolist()

# Get a list of continuous features (based on dtype 'int64' or 'float64')
cont = listing[predictors].select_dtypes(include=['int64', 'float64']).columns.tolist()

# Print the feature names
print("Categorical feature names:", cat)
print("Continuous feature names:", cont)

X = listing[predictors]
y = listing['host_response_time']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_pipe = Pipeline([('onehot', OneHotEncoder(handle_unknown = 'ignore'))])

# Z-Scoring
preprocesser = make_column_transformer((StandardScaler(), cont),
                            (OneHotEncoder(handle_unknown= 'ignore'), cat),
                            remainder = "passthrough")

# Instantiate a LabelEncoder
le = LabelEncoder()

# Fit and transform the target variable
y_train_encoded = le.fit_transform(y_train)  # Apply encoding on training data
y_test_encoded = le.transform(y_test)  # Apply the same transformation to the test data

# Calculate class weights
class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"CLASS WEIGHTS: {class_weight_dict}")

# Create the sample weight array based on class distribution
sample_weights = np.array([class_weight_dict[label] for label in y_train_encoded])

tree = XGBClassifier(
    n_estimators=150, 
    max_depth=5, 
    learning_rate=0.1, 
    min_child_weight=5,
    colsample_bytree=0.8, 
    reg_alpha=0.3, 
    reg_lambda=1.5, 
    subsample=0.8
)

# Fit the model to the data
pipeline = Pipeline([
    ("preprocessor", preprocesser),
    ("tree", tree)
])

print("Fitting the data to the pipeline")

# Fit and Predict
pipeline.fit(X_train, y_train_encoded, tree__sample_weight=sample_weights)

# Predict
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# Convert numeric predictions back to original labels
y_pred_train_labels = le.inverse_transform(y_pred_train)
y_pred_test_labels = le.inverse_transform(y_pred_test)

# Assess performance with classification metrics
print("Listing Performance Metrics: ")

# Accuracy
print("Train Accuracy: ", accuracy_score(y_train_encoded, y_pred_train))
print("Test Accuracy: ", accuracy_score(y_test_encoded, y_pred_test))

# Precision, Recall, and F1-score
print("Train Precision: ", precision_score(y_train_encoded, y_pred_train, average='weighted'))
print("Test Precision: ", precision_score(y_test_encoded, y_pred_test, average = 'weighted'))
print("Train Recall: ", recall_score(y_train_encoded, y_pred_train, average='weighted'))
print("Test Recall: ", recall_score(y_test_encoded, y_pred_test, average='weighted'))
# Compute F1-Score with weighted average for multiclass
print("Train F1-Score: ", f1_score(y_train_encoded, y_pred_train, average='weighted'))
print("Test F1-Score: ", f1_score(y_test_encoded, y_pred_test, average='weighted'))

# Classification Report
print("Classification Report for Train Set: ")
print(classification_report(y_train_encoded, y_pred_train, target_names=["Class 0", "Class 1", "Class 2", "Class 3"]))
print("Classification Report for Test Set: ")
print(classification_report(y_test_encoded, y_pred_test, target_names=["Class 0", "Class 1", "Class 2", "Class 3"]))

# Confusion Matrix
cm_train = confusion_matrix(y_train_encoded, y_pred_train)
cm_test = confusion_matrix(y_test_encoded, y_pred_test)

# Plot confusion matrix for train data
plt.figure(figsize=(6, 5))
sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1", "Class 2", "Class 3"], 
            yticklabels=["Class 0", "Class 1", "Class 2", "Class 3"])
plt.title("Train Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("train_confusion_matrix.png")  # Save as PNG
plt.close()  # Close the plot to avoid display in case of multiple figures

# Plot confusion matrix for test data
plt.figure(figsize=(6, 5))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1", "Class 2", "Class 3"], 
            yticklabels=["Class 0", "Class 1", "Class 2", "Class 3"])
plt.title("Test Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("test_confusion_matrix.png")  # Save as PNG
plt.close()  # Close the plot to avoid display in case of multiple figures

print("Confusion matrix images saved as 'train_confusion_matrix.png' and 'test_confusion_matrix.png'.")

# Save the trained model
joblib.dump(pipeline, 'xgboost_model_noSearch.joblib')
print("Model saved as 'xgboost_model_noSearch.joblib'.")

# Save the trained model
with open('xgboost_model_noSearch.pkl', 'wb') as file:
    pickle.dump(pipeline, file)
print("Model saved as 'xgboost_model_noSearch.pkl'.")

# Compute permutation feature importance
perm_importance = permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=45322)

# Get feature names and their importance values
feature_names = X_train.columns
sorted_idx = perm_importance.importances_mean.argsort()

# Select the top 75 features
top_n = 75
top_features_idx = sorted_idx[-top_n:]  # Get indices of the top 75 features

# Create a DataFrame for the top features and their permutation importance
top_features_df = pd.DataFrame({
    'Feature': feature_names[top_features_idx],
    'Permutation Importance': perm_importance.importances_mean[top_features_idx]
})

# Sort the DataFrame by importance
top_features_df = top_features_df.sort_values(by='Permutation Importance', ascending=False)

# Display the DataFrame
print("Listing Top Features:")
print(top_features_df)
