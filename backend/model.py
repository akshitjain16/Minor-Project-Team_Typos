# %% [markdown]
# # Import Libraries

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# # Add Dataset path

# %%
folder_path = 'D:/Projects/Android-Malware-Detection-System-Using-Machine-Learning/'
plots_path = folder_path + 'Plots/'
dataset_path = folder_path + 'Dataset/'
dataset_file = dataset_path + 'Android_Permission.csv'
unsampled = 'Unsampled'
oversampled = 'Oversampled'
undersampled = 'Undersampled'

# %%
np.random.seed(0)

# %% [markdown]
# # Import Dataset

# %%
df = pd.read_csv(dataset_file, sep=',')
df

# %%
df.shape

# %%
df.columns

# %%
df.describe()

# %%
plt.rcParams["figure.figsize"] = (10,5)
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# %%
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# %%
def plot_missing_values(null_columns) :
    plt.xlabel("Column Names")
    plt.ylabel("Missing values")
    plt.xticks(rotation = 30, ha = "right")
    name = 'Columns Name vs Missing Values'
    plt.tight_layout()
    plt.title(name)
    ax = sns.barplot(x = list(null_columns.keys()), y = list(null_columns.values()))
    ax.bar_label(ax.containers[0])
    plt.savefig( plots_path + name + ".png", bbox_inches = 'tight')
    plt.show()

# %%
null_sum = dict(df.isna().sum()[df.isna().sum()>0])
plot_missing_values(null_sum)

# %%
def replace_mean(x):
    if x.dtype!='object':
        return x.fillna(x.mean())
    return x

# %%
new_df = df.apply(lambda x: replace_mean(x),axis=1)
new_df

# %%
df_permission = pd.concat([new_df['App'],new_df.iloc[:,10:]],axis=1)
df_permission

# %%
missing_values_column = df_permission.columns[df_permission.isna().any()]
for c in missing_values_column :
    if df_permission[c].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] :
        df_permission[c].replace(np.nan, np.nanmean(df_permission[c].unique()), inplace=True)


# %%
def plot_class_distribution(vals, sample_type) :
    _ , data = np.unique(vals, return_counts = True)
    labels = ['Benign','Malware']
    colors = sns.color_palette('bright')[0:5]
    plt.tight_layout()
    plt.pie(data, labels = labels, colors = colors, autopct = lambda p: '{:.2f}%\n({:.0f})'.format(p, (p/100) * data.sum()))
    name = 'Class Distribution'
    plt.title(name + ' - ' + sample_type, color = 'red')
    plt.savefig(folder_path + 'Data' + '/' + sample_type + '/' + name + '.png', bbox_inches='tight')
    plt.show()

# %%
plot_class_distribution(df_permission['Class'], unsampled)

# %%
df_permission

# %%
df_permission.index = df_permission['App']
df_permission.drop('App', inplace = True, axis = 1)
df_permission

# %%
x = df_permission.iloc[1: , : -1]
y = df_permission.iloc[1: , -1]
np.random.seed(0)

# %%
x

# %%


# %% [markdown]
# ## EDA
# * Exploratory data analysis uses statistical summaries and graphical representations to evaluate the data and find trends, patterns, or verify assumptions in the data.
# * EDA helps with a better understanding of the variables in the data collection and their relationships, and is usually used to investigate what data might disclose beyond the formal modeling or hypothesis testing assignment.
# * It can also assist in determining the suitability of the statistical methods you are contemplating using for data analysis.
# 

# %%
# Check for missing values
print(df.isnull().sum())

# %%
# Check the correlation between the features
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), cmap='coolwarm')
plt.title("Correlation Between Features")
plt.show()


# %%
data = df

# %%
# Analyze the distribution of dangerous and safe permissions count:

import matplotlib.pyplot as plt

dangerous_permissions = data['Dangerous permissions count']
safe_permissions = data['Safe permissions count']

plt.hist([dangerous_permissions, safe_permissions], label=['Dangerous', 'Safe'], bins=20, alpha=0.7)
plt.title('Distribution of Dangerous and Safe Permissions Count')
plt.xlabel('Permissions Count')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()


# %%
# Analyze the relationship between app ratings and permissions count:

import seaborn as sns

sns.scatterplot(x=data['Rating'], y=data['Dangerous permissions count'], label='Dangerous')
sns.scatterplot(x=data['Rating'], y=data['Safe permissions count'], label='Safe')
plt.title('Relationship between App Ratings and Permissions Count')
plt.xlabel('App Rating')
plt.ylabel('Permissions Count')
plt.legend(loc='upper right')
plt.show()


# %%
# Analyze the distribution of app ratings:

sns.histplot(data['Rating'], kde=True)
plt.title('Distribution of App Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()


# %%
# Analyze the distribution of app categories:

category_counts = data['Category'].value_counts()
category_counts.plot(kind='bar', figsize=(20, 10))
plt.title('Distribution of App Categories')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.show()


# %%
# Analyze the relationship between the number of ratings and app ratings:

import seaborn as sns

sns.scatterplot(x=data['Number of ratings'], y=data['Rating'])
# plt.figure(figsize=(10, 6))
plt.title('Relationship between Number of Ratings and App Ratings')
plt.xlabel('Number of Ratings')
plt.ylabel('App Rating')
plt.show()


# %%
# Analyze the relationship between app price and permissions count:

# Convert price to numeric values
data['Price_numeric'] = data['Price'].apply(lambda x: float(x[1:]) if isinstance(x, str) and x.startswith('$') else 0)

sns.scatterplot(x=data['Price_numeric'], y=data['Dangerous permissions count'], label='Dangerous')
sns.scatterplot(x=data['Price_numeric'], y=data['Safe permissions count'], label='Safe')
# plt.figure(figsize=(10, 6))
plt.title('Relationship between App Price and Permissions Count')
plt.xlabel('App Price')
plt.ylabel('Permissions Count')
plt.legend(loc='upper right')
plt.show()



# %%
# Analyze the average permissions count per category:

import numpy as np

category_dangerous_mean = data.groupby('Category')['Dangerous permissions count'].mean()
category_safe_mean = data.groupby('Category')['Safe permissions count'].mean()

categories = data['Category'].unique()
x_bar = np.arange(len(categories))
width = 0.4

fig, ax = plt.subplots(figsize=(20, 10))
rects1 = ax.bar(x_bar - width/2, category_dangerous_mean, width, label='Dangerous')
rects2 = ax.bar(x_bar + width/2, category_safe_mean, width, label='Safe')

ax.set_title('Average Permissions Count per Category')
ax.set_xlabel('Category')
ax.set_ylabel('Average Permissions Count')
ax.set_xticks(x_bar)
ax.set_xticklabels(categories, rotation=45)
ax.legend()

plt.show()


# %%
# Analyze the distribution of each feature

# Get the list of features
features = data.columns.tolist()

# Remove non-numeric features
features.remove('App')
features.remove('Package')
features.remove('Category')
features.remove('Description')
features.remove('Related apps')

count = 0
# Plot the distribution of each feature
for feature in features:
    if count < 5:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[feature], kde=True)
        plt.title('Distribution of ' + feature)
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()
        count = count+1


# %%
# Analyze the distribution of the number of API calls per permission:

# Group the data by permission and sum the number of API calls
api_calls = data.groupby('Description')['Number of ratings'].sum()

# Plot the distribution of the number of API calls per permission
plt.figure(figsize=(10, 6))
plt.hist(api_calls, bins=20)
plt.title('Distribution of API Calls per Permission')
plt.xlabel('API Calls')
plt.ylabel('Frequency')
plt.show()


# %% [markdown]
# # Test Model

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier,export_graphviz

# %%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %%
def logistic(X_train, X_test, y_train, y_test):
    reg=LogisticRegression(max_iter=200,n_jobs=-1).fit(X_train,y_train)
    y_predict=reg.predict(X_test)
    y_predict_train=reg.predict(X_train)
    precision_train = precision_score(y_train, y_predict_train)
    precision = precision_score(y_test, y_predict)
    accuracy = accuracy_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    roc_auc=roc_auc_score(y_test, y_predict)
    print(f"Training Accuracy %.2f"%(precision_train))
    print(f"Test Accuracy %.2f"%(precision))
    print(f"Recall Score %.2f"%(recall))
    print(f'ROC Score %.2f'%(roc_auc))
    return reg

# %%
logistic(X_train, X_test, y_train, y_test)

# %%
def Naive(X_train, X_test, y_train, y_test):
    reg=GaussianNB().fit(X_train,y_train)
    y_predict=reg.predict(X_test)
    y_predict_train=reg.predict(X_train)
    precision_train = precision_score(y_train, y_predict_train)
    precision = precision_score(y_test, y_predict)
    accuracy = accuracy_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    roc_auc=roc_auc_score(y_test, y_predict)
    print(f"Training Accuracy %.2f"%(precision_train))
    print(f"Test Accuracy %.2f"%(precision))
    print(f"Recall Score %.2f"%(recall))
    print(f'ROC Score %.2f'%(roc_auc))
    return reg

# %%
Naive(X_train, X_test, y_train, y_test)

# %%
from io import StringIO
import pydotplus
from IPython.display import Image

# %%
def visualize_dtree(reg, columns, sample_type, name = 'dt') :
    dot_data = StringIO()
    export_graphviz(reg, out_file = dot_data,  
                    filled = True, rounded=True,
                    special_characters = True,feature_names = columns, class_names = ['Benign','Malware'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.create_png())
    graph.write_png(folder_path + '/' + 'Data' + '/' + sample_type + '/' + name + '.png')

# %%
def dtree(X_train, X_test, y_train, y_test):
    reg=DecisionTreeClassifier(max_features='sqrt',max_depth=10,max_leaf_nodes=10).fit(X_train,y_train)
    y_predict=reg.predict(X_test)
    y_predict_train=reg.predict(X_train)
    precision_train = precision_score(y_train, y_predict_train)
    precision = precision_score(y_test, y_predict)
    accuracy = accuracy_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    roc_auc=roc_auc_score(y_test, y_predict)
    print(f"Training Accuracy %.2f"%(precision_train))
    print(f"Test Accuracy %.2f"%(precision))
    print(f"Recall Score %.2f"%(recall))
    print(f'ROC Score %.2f'%(roc_auc))
    return reg

# %%
reg = dtree(X_train, X_test, y_train, y_test)
visualize_dtree(reg, df_permission.columns[:-1], unsampled)

# %% [markdown]
# # Confusion Matrix using different techniques

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train and evaluate logistic regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)

lr_cm = confusion_matrix(y_test, lr_y_pred)
lr_cr = classification_report(y_test, lr_y_pred)

# Train and evaluate decision tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_y_pred = dt_model.predict(X_test)

dt_cm = confusion_matrix(y_test, dt_y_pred)
dt_cr = classification_report(y_test, dt_y_pred)

# Train and evaluate random forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

rf_cm = confusion_matrix(y_test, rf_y_pred)
rf_cr = classification_report(y_test, rf_y_pred)

# Train and evaluate neural network model
nn_model = MLPClassifier(random_state=42)
nn_model.fit(X_train, y_train)
nn_y_pred = nn_model.predict(X_test)

nn_cm = confusion_matrix(y_test, nn_y_pred)
nn_cr = classification_report(y_test, nn_y_pred)

# Plot confusion matrix for each model
sns.heatmap(lr_cm, annot=True, fmt='g', cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

sns.heatmap(dt_cm, annot=True, fmt='g', cmap='Blues')
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

sns.heatmap(rf_cm, annot=True, fmt='g', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

sns.heatmap(nn_cm, annot=True, fmt='g', cmap='Blues')
plt.title("Neural Network Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Print classification report for each model
print("Logistic Regression:\n", lr_cr)
print("Decision Tree:\n", dt_cr)
print("Random Forest:\n", rf_cr)
print("Neural Network:\n", nn_cr)


# %%
# Calculate accuracy
lr_accuracy = accuracy_score(y_test, lr_y_pred)
dt_accuracy = accuracy_score(y_test, dt_y_pred)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
nn_accuracy = accuracy_score(y_test, nn_y_pred)

# %%
import matplotlib.pyplot as plt

# Plot the accuracies
models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Neural Network']
accuracies = [lr_accuracy, dt_accuracy, rf_accuracy, nn_accuracy]

plt.bar(models, accuracies)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.show()

# %% [markdown]
# # Feature Selection:
# * It will then train a linear regression model with recursive feature elimination (RFE) to select the two most important features for predicting the app rating and print the selected features.

# %%
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Select the features and target variable
X = df[['Number of ratings', 'Dangerous permissions count', 'Safe permissions count']]
y = df['Rating']

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_imputed = imputer.fit_transform(X)
y_imputed = imputer.fit_transform(y.values.reshape(-1,1)).ravel()

# Train a linear regression model with RFE
lr = LinearRegression()
rfe = RFE(lr, n_features_to_select=2)
rfe.fit(X_imputed, y_imputed)

# Print the selected features
print('Selected Features:', X.columns[rfe.support_])

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Select the features and target variable
X = df[['Dangerous permissions count', 'Safe permissions count']]
y = df['Rating'].apply(lambda x: 1 if x >= 4 else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train and evaluate a logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Logistic Regression Accuracy:', acc)

# Train and evaluate a decision tree classification model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Decision Tree Classification Accuracy:', acc)

# Train and evaluate a random forest classification model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Random Forest Classification Accuracy:', acc)


# %%
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Select the features and target variable
X = df[['Number of ratings', 'Dangerous permissions count', 'Safe permissions count']]
y = df['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train and evaluate a linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Linear Regression MSE:', mse)

# Train and evaluate a decision tree regression model
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Decision Tree Regression MSE:', mse)

# Train and evaluate a random forest regression model
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Random Forest Regression MSE:', mse)


# %% [markdown]
# # Clustering

# %%
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select the features
X = df[['Number of ratings', 'Dangerous permissions count', 'Safe permissions count']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# Add the cluster labels to the dataframe
df['Cluster'] = labels

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['Number of ratings'], df['Dangerous permissions count'], c=df['Cluster'])
plt.xlabel('Number of ratings')
plt.ylabel('Dangerous permissions count')
plt.title('K-means Clustering Results')
plt.show()


# %% [markdown]
# # Dimensionality Reduction:
# * We are using dimensionality reduction techniques like PCA to reduce the number of features in the dataset and visualize the relationships between the remaining features.

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Select the features
X = df[['Number of ratings', 'Dangerous permissions count', 'Safe permissions count']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Perform PCA
pca = PCA(n_components=2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

# Plot the PCA results
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Rating'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Results')
plt.show()


# %% [markdown]
# # Anomaly Detection:
# * We are using anomaly detection techniques like isolation forest or one-class SVM to identify apps that have unusual combinations of features or permissions.

# %%
from sklearn.ensemble import IsolationForest

# Select the features
X = df[['Number of ratings', 'Dangerous permissions count', 'Safe permissions count']]

# Detect anomalies with Isolation Forest
iforest = IsolationForest(contamination=0.05, random_state=42)
iforest.fit(X)
labels = iforest.predict(X)

# Add the anomaly labels to the dataframe
df['Anomaly'] = labels

# Print the number of anomalies
print('Number of Anomalies:', len(df[df['Anomaly'] == -1]))


# %% [markdown]
# # Sampling
# 

# %%
# !pip install imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# %%
x = df_permission.iloc[1: , : -1]
y = df_permission.iloc[1: , -1]
y = y.astype(int)
oversample = RandomOverSampler(sampling_strategy=0.9)
X_over, y_over = oversample.fit_resample(x, y)
X_over, y_over

# %%
plot_class_distribution(y_over, oversampled)

# %%
X_over_train, X_over_test, y_over_train, y_over_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42)

# %%
logistic(X_over_train, X_over_test, y_over_train, y_over_test)

# %%
Naive(X_over_train, X_over_test, y_over_train, y_over_test)

# %%
reg_over = dtree(X_over_train, X_over_test, y_over_train, y_over_test)
visualize_dtree(reg_over, df_permission.columns[:-1], oversampled)

# %%
undersample = RandomUnderSampler(sampling_strategy=0.9)
X_under, y_under = undersample.fit_resample(x, y)

# %%
plot_class_distribution(y_under, undersampled)

# %%
X_under_train, X_under_test, y_under_train, y_under_test = train_test_split(X_under, y_under, test_size=0.2, random_state=42)

# %%
logistic(X_under_train, X_under_test, y_under_train, y_under_test)

# %%
Naive(X_under_train, X_under_test, y_under_train, y_under_test)

# %%
reg_under = dtree(X_under_train, X_under_test, y_under_train, y_under_test)
visualize_dtree(reg_under, df_permission.columns[:-1], undersampled)

# %%
# from imblearn.over_sampling import SMOTE

# smotesample = SMOTE(sampling_strategy=0.9)
# X_smote, y_smote = smotesample.fit_resample(x, y)

# plot_class_distribution(y_smote, smotesampled)

# X_smote_train, X_smote_test, y_smote_train, y_smote_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

# logistic(X_smote_train, X_smote_test, y_smote_train, y_smote_test)

# Naive(X_smote_train, X_smote_test, y_smote_train, y_smote_test)

# reg_smote = dtree(X_smote_train, X_smote_test, y_smote_train, y_smote_test)
# visualize_dtree(reg_smote, smotesampled)

# %% [markdown]
# ## PCA
# * Principal Component Analysis (PCA) is a well-known unsupervised learning approach for decreasing data dimensionality.
# * While minimizing information loss, it simultaneously improves interpretability.
# * It also assists in identifying the most important features in a dataset and makes the data easier to plot in 2D and 3D.
# 

# %%
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Select the columns for PCA
columns = ['Rating', 'Number of ratings', 'Dangerous permissions count', 'Safe permissions count']

# Replace missing values with the mean of the column
imputer = SimpleImputer(strategy='mean')
df[columns] = imputer.fit_transform(df[columns])

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[columns])

# Perform PCA
pca = PCA(n_components=2)
pca.fit(data_scaled)
data_pca = pca.transform(data_scaled)

# Create a new dataframe with the PCA results
df_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])

# Add the original columns to the new dataframe
df_pca['Rating'] = df['Rating']
df_pca['Number of ratings'] = df['Number of ratings']
df_pca['Dangerous permissions count'] = df['Dangerous permissions count']
df_pca['Safe permissions count'] = df['Safe permissions count']

# Plot the results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Rating', data=df_pca)
plt.title('PCA Results')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


# %%
pca = PCA(n_components=100)
principalComponents = pca.fit_transform(df_permission)
features = range(pca.n_components_)
PCA_components = pd.DataFrame(principalComponents)
ratios = pca.explained_variance_ratio_*100
PCA_components

# %%
def plot_pca(features, ratios):
    plt.figure(figsize=(10, 6))
    plt.xlabel('PCA features')
    plt.ylabel('variance Percentage')
    plt.xticks(features, rotation=70)
    name = 'PCA features vs Variance Percentage'
    plt.title(name)
    plt.tight_layout()
    plt.bar(features, ratios, color=sns.color_palette("flare"))
    plt.savefig(plots_path + name + '.png', bbox_inches = 'tight')
    plt.show()

# %%
plot_pca(features, ratios)

# %%
temp_df=pd.DataFrame(pca.inverse_transform(principalComponents))
temp_df

# %%
temp_df[temp_df<=0]=0
temp_df[temp_df!=0]=1

# %%
temp_df=temp_df.astype(int)
temp_df.index=df['App']
temp_df

# %%
df_y=df['Class']
df_y.index=df['App']

# %%
df_feature=pd.concat([temp_df.iloc[:,:10],df_y],axis=1)
df_feature

# %%
X_feature=df_feature.iloc[:,:-1]
y_feature=df_feature.iloc[:,-1]

# %%
X_feature_train, X_feature_test, y_feature_train, y_feature_test = train_test_split(X_feature, y_feature, test_size=0.2, random_state=42)
X_feature_train.shape,X_feature_test.shape,y_feature_train.shape,y_feature_test.shape

# %%
logistic(X_feature_train, X_feature_test, y_feature_train, y_feature_test)

# %%
Naive(X_feature_train, X_feature_test, y_feature_train, y_feature_test)

# %%
reg_feature = dtree(X_feature_train, X_feature_test, y_feature_train, y_feature_test)
visualize_dtree(reg_feature, [str(c) for c in df_feature.columns[:-1]], unsampled, 'pca_dt')

# %%
from sklearn.model_selection import GridSearchCV

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
def randomforest(X_train, X_test, y_train, y_test) :
    reg = RandomForestClassifier(n_estimators=200, n_jobs = -1).fit(X_train,y_train)
    y_predict=reg.predict(X_test)
    y_predict_train=reg.predict(X_train)
    precision_train = precision_score(y_train, y_predict_train)
    precision = precision_score(y_test, y_predict)
    accuracy = accuracy_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    roc_auc=roc_auc_score(y_test, y_predict)
    print(f"Training Accuracy %.2f"%(precision_train))
    print(f"Test Accuracy %.2f"%(precision))
    print(f"Recall Score %.2f"%(recall))
    print(f'ROC Score %.2f'%(roc_auc))
    return reg

# %%
randomforest(X_feature_train, X_feature_test, y_feature_train, y_feature_test)

# %%
import xgboost as xgb
import joblib

# %%
def xgboost(X_train, X_test, y_train, y_test) :
    reg = GridSearchCV(estimator = xgb.XGBClassifier(scale_pos_weight = 0.5, n_jobs = -1), param_grid={}, cv = 10)
    reg = reg.fit(X_train,y_train)
    y_predict=reg.predict(X_test)
    y_predict_train=reg.predict(X_train)
    precision_train = precision_score(y_train, y_predict_train)
    precision = precision_score(y_test, y_predict)
    accuracy = accuracy_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    roc_auc=roc_auc_score(y_test, y_predict)
    print(f"Training Accuracy %.2f"%(precision_train))
    print(f"Test Accuracy %.2f"%(precision))
    print(f"Recall Score %.2f"%(recall))
    print(f'ROC Score %.2f'%(roc_auc))
    joblib.dump(reg, 'xgboost_model.joblib')
    return reg

# %%
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(x, y, test_size=0.2, random_state=42)
xgboost(X_train_cls, X_test_cls, y_train_cls, y_test_cls)

# %%
xgboost(X_feature_train, X_feature_test, y_feature_train, y_feature_test)

# %%
from sklearn import svm

# %%
def svm_classfier(X_train, X_test, y_train, y_test) :
    reg = svm.SVC().fit(X_train,y_train)
    y_predict=reg.predict(X_test)
    y_predict_train=reg.predict(X_train)
    precision_train = precision_score(y_train, y_predict_train)
    precision = precision_score(y_test, y_predict)
    accuracy = accuracy_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    roc_auc=roc_auc_score(y_test, y_predict)
    print(f"Training Accuracy %.2f"%(precision_train))
    print(f"Test Accuracy %.2f"%(precision))
    print(f"Recall Score %.2f"%(recall))
    print(f'ROC Score %.2f'%(roc_auc))
    return reg

# %%
svm_classfier(X_feature_train, X_feature_test, y_feature_train, y_feature_test)

# %%
from sklearn.neural_network import MLPClassifier

# %%
def mlp(X_train, X_test, y_train, y_test) :
    reg = MLPClassifier(random_state = 42, max_iter = 300).fit(X_train,y_train)
    y_predict=reg.predict(X_test)
    y_predict_train=reg.predict(X_train)
    precision_train = precision_score(y_train, y_predict_train)
    precision = precision_score(y_test, y_predict)
    accuracy = accuracy_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    roc_auc=roc_auc_score(y_test, y_predict)
    print(f"Training Accuracy %.2f"%(precision_train))
    print(f"Test Accuracy %.2f"%(precision))
    print(f"Recall Score %.2f"%(recall))
    print(f'ROC Score %.2f'%(roc_auc))
    return reg

# %%
mlp(X_feature_train, X_feature_test, y_feature_train, y_feature_test)

# %%
# Load the model from the file
loaded_model = joblib.load('xgboost_model.joblib')

# Use the loaded model to make predictions
loaded_model.predict(X_feature_test)


