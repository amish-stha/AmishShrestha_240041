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

# Downsample majority class to minority class size
non_fraud_downsampled = resample(
    non_fraud,
    replace=False,
    n_samples=len(fraud),
    random_state=42
)

# Combine balanced data
balanced_train = pd.concat([fraud, non_fraud_downsampled])

# Shuffle the dataset
balanced_train = balanced_train.sample(frac=1, random_state=42)

# Split features and target again
X_sampled = balanced_train.drop('is_fraud', axis=1)
y_sampled = balanced_train['is_fraud']

# Scale the sampled train features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_sampled)

# Preprocess test data the same way as train data
test_df = test_df.drop([
    'Unnamed: 0', 'trans_num', 'unix_time', 'cc_num', 'first', 'last',
    'street', 'merchant', 'city', 'state', 'zip', 'trans_date_trans_time'
], axis=1)

test_df['dob'] = pd.to_datetime(test_df['dob'], errors='coerce')
test_df['age'] = 2019 - test_df['dob'].dt.year
test_df = test_df.drop('dob', axis=1)

test_df = pd.get_dummies(test_df, columns=['category', 'gender', 'job'], drop_first=True)

# Align test columns with train columns, filling missing columns with 0
test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

# Separate features and target from test data
X_test = test_df.drop('is_fraud', axis=1)
y_test = test_df['is_fraud']

# Scale test features using scaler fitted on train data
X_test_scaled = scaler.transform(X_test)

# Create and train KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_sampled)

# Predict on test data
y_pred = knn.predict(X_test_scaled)

# Evaluate model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

