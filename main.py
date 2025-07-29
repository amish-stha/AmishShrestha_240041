import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns


# Load train and test datasets
train_df = pd.read_csv('fraudTrain.csv')
test_df = pd.read_csv('fraudTest.csv')

# Drop unnecessary columns from train data
train_df = train_df.drop([
    'Unnamed: 0', 'trans_num', 'unix_time', 'cc_num', 'first', 'last',
    'street', 'merchant', 'city', 'state', 'zip', 'trans_date_trans_time'
], axis=1)

# Convert dob to datetime and calculate age
train_df['dob'] = pd.to_datetime(train_df['dob'], errors='coerce')
train_df['age'] = 2019 - train_df['dob'].dt.year
train_df = train_df.drop('dob', axis=1)

# One-hot encode categorical variables
train_df = pd.get_dummies(train_df, columns=['category', 'gender', 'job'], drop_first=True)

# Separate features and target from train data
X_train = train_df.drop('is_fraud', axis=1)
y_train = train_df['is_fraud']

# Downsample train data to 50,000 rows preserving class distribution

# Combine features and target for easier resampling
train_data = pd.concat([X_train, y_train], axis=1)

# Separate majority and minority classes
fraud = train_data[train_data['is_fraud'] == 1]
non_fraud = train_data[train_data['is_fraud'] == 0]


