import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from io import StringIO
import pydotplus
import xgboost as xgb
import joblib
from sklearn import svm
import os

BASE_DIR = './'
FIGURES_DIR = 'Generated_Figures/'
DATA_DIR = 'Dataset/'
INPUT_DATA_FILE = os.path.join(DATA_DIR, 'Android_Permission.csv')
RAW = 'Raw'
BALANCED_OVER = 'OverSampled'
BALANCED_UNDER = 'UnderSampled'

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(os.path.join('Output', RAW), exist_ok=True)
os.makedirs(os.path.join('Output', BALANCED_OVER), exist_ok=True)
os.makedirs(os.path.join('Output', BALANCED_UNDER), exist_ok=True)

np.random.seed(42)

source_df = pd.read_csv(INPUT_DATA_FILE, sep=',')

plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200

import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

def create_bar_graph(dictionary, x_axis_label, y_axis_label, graph_title, save_path):
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.title(graph_title)
    the_bar_plot = sns.barplot(x=list(dictionary.keys()), y=list(dictionary.values()))
    the_bar_plot.bar_label(the_bar_plot.containers[0])
    plt.savefig(save_path + graph_title + ".jpg", bbox_inches='tight')
    plt.show()

null_value_counts = dict(source_df.isna().sum()[source_df.isna().sum() > 0])
create_bar_graph(null_value_counts, "Feature", "Null Count", "Count of Nulls by Feature", FIGURES_DIR)

def fill_missing_with_mean(a_column):
    if a_column.dtype != 'object':
        return a_column.fillna(a_column.mean())
    return a_column

processed_df = source_df.apply(lambda c: fill_missing_with_mean(c), axis=0)

attributes_df = pd.concat([processed_df['App'], processed_df.iloc[:, 10:]], axis=1)

for feature_name in attributes_df.columns[attributes_df.isna().any()]:
    if attributes_df[feature_name].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
        attributes_df[feature_name].replace(np.nan, np.nanmean(attributes_df[feature_name].unique()), inplace=True)

def create_pie_chart(target_values, sample_category):
    _, value_counts = np.unique(target_values, return_counts=True)
    target_labels = ['Type A', 'Type B']
    chart_colors = sns.color_palette('pastel')[0:5]
    plt.tight_layout()
    plt.pie(value_counts, labels=target_labels, colors=chart_colors, autopct=lambda p: f'{(p):.2f}%({(p/100*value_counts.sum()):.0f}')
    chart_main_title = 'Target Distribution'
    plt.title(chart_main_title + ' - ' + sample_category, color='blue')
    plt.savefig(os.path.join('Output', sample_category, chart_main_title + '.jpg'), bbox_inches='tight')
    plt.show()

create_pie_chart(attributes_df['Class'], RAW)

attributes_df.index = attributes_df['App']
attributes_df.drop('App', inplace=True, axis=1)

x_data = attributes_df.iloc[1:, :-1]
y_data = attributes_df.iloc[1:, -1]
np.random.seed(42)

print(source_df.isnull().sum())

numeric_df_for_correlation = source_df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df_for_correlation.corr(), cmap='viridis')
plt.title("Feature Correlation Matrix")
plt.show()

high_risk_permission_count = source_df['Dangerous permissions count']
low_risk_permission_count = source_df['Safe permissions count']

plt.hist([high_risk_permission_count, low_risk_permission_count], label=['High-Risk', 'Low-Risk'], bins=25, alpha=0.6)
plt.title('Histogram of Permission Counts')
plt.xlabel('Number of Permissions')
plt.ylabel('Count')
plt.legend(loc='upper center')
plt.show()

sns.scatterplot(x=source_df['Rating'], y=source_df['Dangerous permissions count'], label='High-Risk')
sns.scatterplot(x=source_df['Rating'], y=source_df['Safe permissions count'], label='Low-Risk')
plt.title('App Rating vs. Permission Counts')
plt.xlabel('Application Rating')
plt.ylabel('Number of Permissions')
plt.legend(loc='upper center')
plt.show()

sns.histplot(source_df['Rating'], kde=True, color='purple')
plt.title('Density of Application Ratings')
plt.xlabel('Rating Value')
plt.ylabel('Density')
plt.show()

application_category_counts = source_df['Category'].value_counts()
application_category_counts.plot(kind='barh', figsize=(15, 12))
plt.title('Frequency of App Categories')
plt.xlabel('Frequency')
plt.ylabel('Category Name')
plt.show()

sns.scatterplot(x=source_df['Number of ratings'], y=source_df['Rating'], color='green')
plt.title('Number of Ratings vs. App Rating')
plt.xlabel('Total Ratings')
plt.ylabel('Average Rating')
plt.show()

source_df['Price_numeric'] = source_df['Price'].apply(lambda val: float(val[1:]) if isinstance(val, str) and val.startswith('$') else 0)

sns.scatterplot(x=source_df['Price_numeric'], y=source_df['Dangerous permissions count'], label='High-Risk')
sns.scatterplot(x=source_df['Price_numeric'], y=source_df['Safe permissions count'], label='Low-Risk')
plt.title('App Price vs. Permission Counts')
plt.xlabel('Application Price (USD)')
plt.ylabel('Number of Permissions')
plt.legend(loc='upper right')
plt.show()

avg_high_risk_perms_by_category = source_df.groupby('Category')['Dangerous permissions count'].mean()
avg_low_risk_perms_by_category = source_df.groupby('Category')['Safe permissions count'].mean()

category_names = source_df['Category'].unique()
x_positions = np.arange(len(category_names))
group_bar_width = 0.35

figure, axis = plt.subplots(figsize=(22, 12))
bar_group1 = axis.bar(x_positions - group_bar_width / 2, avg_high_risk_perms_by_category, group_bar_width, label='High-Risk')
bar_group2 = axis.bar(x_positions + group_bar_width / 2, avg_low_risk_perms_by_category, group_bar_width, label='Low-Risk')

axis.set_title('Average Permission Count by Category')
axis.set_xlabel('Application Category')
axis.set_ylabel('Average Permission Count')
axis.set_xticks(x_positions)
axis.set_xticklabels(category_names, rotation=90)
axis.legend()
plt.show()

features_for_distribution_plot = source_df.columns.tolist()
features_for_distribution_plot.remove('App')
features_for_distribution_plot.remove('Package')
features_for_distribution_plot.remove('Category')
features_for_distribution_plot.remove('Description')
features_for_distribution_plot.remove('Related apps')

graph_plot_counter = 0
for feature in features_for_distribution_plot:
    if graph_plot_counter < 5:
        plt.figure(figsize=(10, 6))
        sns.histplot(source_df[feature], kde=True, color='orange')
        plt.title('Distribution of ' + feature)
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()
        graph_plot_counter += 1

api_calls_by_permission = source_df.groupby('Description')['Number of ratings'].sum()
plt.figure(figsize=(10, 6))
plt.hist(api_calls_by_permission, bins=20, color='cyan')
plt.title('Histogram of API Calls per Permission')
plt.xlabel('API Calls')
plt.ylabel('Frequency')
plt.show()

X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(x_data, y_data, test_size=0.25, random_state=42)

def print_model_metrics(ml_model, train_x, test_x, train_y, test_y):
    pred_y_test = ml_model.predict(test_x)
    pred_y_train = ml_model.predict(train_x)
    p_train = precision_score(train_y, pred_y_train)
    p_test = precision_score(test_y, pred_y_test)
    acc_test = accuracy_score(test_y, pred_y_test)
    rec_test = recall_score(test_y, pred_y_test)
    roc_auc_test = roc_auc_score(test_y, pred_y_test)
    print(f"Train Precision: {p_train:.3f}")
    print(f"Test Precision: {p_test:.3f}")
    print(f"Test Accuracy: {acc_test:.3f}")
    print(f"Test Recall: {rec_test:.3f}")
    print(f'Test ROC AUC: {roc_auc_test:.3f}')

def execute_logistic_regression(train_x, test_x, train_y, test_y):
    the_model = LogisticRegression(max_iter=250, n_jobs=-1).fit(train_x, train_y)
    print_model_metrics(the_model, train_x, test_x, train_y, test_y)
    return the_model

def execute_gaussian_nb(train_x, test_x, train_y, test_y):
    the_model = GaussianNB().fit(train_x, train_y)
    print_model_metrics(the_model, train_x, test_x, train_y, test_y)
    return the_model

execute_logistic_regression(X_train_main, X_test_main, y_train_main, y_test_main)
execute_gaussian_nb(X_train_main, X_test_main, y_train_main, y_test_main)

def save_tree_visualization(ml_model, columns, sample_cat, f_name='tree_viz'):
    tree_buffer = StringIO()
    export_graphviz(ml_model, out_file=tree_buffer, filled=True, rounded=True,
                    special_characters=True, feature_names=columns, class_names=['Type A', 'Type B'])
    the_graph = pydotplus.graph_from_dot_data(tree_buffer.getvalue())
    the_graph.write_png(os.path.join('Output', sample_cat, f_name + '.png'))

def execute_decision_tree(train_x, test_x, train_y, test_y):
    the_model = DecisionTreeClassifier(max_features='log2', max_depth=12, max_leaf_nodes=15).fit(train_x, train_y)
    print_model_metrics(the_model, train_x, test_x, train_y, test_y)
    return the_model

main_dt_model = execute_decision_tree(X_train_main, X_test_main, y_train_main, y_test_main)
save_tree_visualization(main_dt_model, attributes_df.columns[:-1], RAW)

log_reg_model = LogisticRegression(random_state=42).fit(X_train_main, y_train_main)
log_reg_preds = log_reg_model.predict(X_test_main)
log_reg_cm = confusion_matrix(y_test_main, log_reg_preds)
log_reg_report = classification_report(y_test_main, log_reg_preds)

dec_tree_model = DecisionTreeClassifier(random_state=42).fit(X_train_main, y_train_main)
dec_tree_preds = dec_tree_model.predict(X_test_main)
dec_tree_cm = confusion_matrix(y_test_main, dec_tree_preds)
dec_tree_report = classification_report(y_test_main, dec_tree_preds)

rand_forest_model = RandomForestClassifier(random_state=42).fit(X_train_main, y_train_main)
rand_forest_preds = rand_forest_model.predict(X_test_main)
rand_forest_cm = confusion_matrix(y_test_main, rand_forest_preds)
rand_forest_report = classification_report(y_test_main, rand_forest_preds)

neural_net_model = MLPClassifier(random_state=42, max_iter=600).fit(X_train_main, y_train_main)
neural_net_preds = neural_net_model.predict(X_test_main)
neural_net_cm = confusion_matrix(y_test_main, neural_net_preds)
neural_net_report = classification_report(y_test_main, neural_net_preds)

def create_confusion_matrix_plot(cm, a_title):
    sns.heatmap(cm, annot=True, fmt='d', cmap='cividis')
    plt.title(a_title)
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.show()

create_confusion_matrix_plot(log_reg_cm, "Logistic Regression CM")
create_confusion_matrix_plot(dec_tree_cm, "Decision Tree CM")
create_confusion_matrix_plot(rand_forest_cm, "Random Forest CM")
create_confusion_matrix_plot(neural_net_cm, "Neural Network CM")

print("Logistic Regression Classification Report:\n", log_reg_report)
print("Decision Tree Classification Report:\n", dec_tree_report)
print("Random Forest Classification Report:\n", rand_forest_report)
print("Neural Network Classification Report:\n", neural_net_report)

log_reg_accuracy = accuracy_score(y_test_main, log_reg_preds)
dec_tree_accuracy = accuracy_score(y_test_main, dec_tree_preds)
rand_forest_accuracy = accuracy_score(y_test_main, rand_forest_preds)
neural_net_accuracy = accuracy_score(y_test_main, neural_net_preds)

model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Neural Network']
model_accuracies = [log_reg_accuracy, dec_tree_accuracy, rand_forest_accuracy, neural_net_accuracy]

plt.bar(model_names, model_accuracies, color=['red', 'green', 'blue', 'purple'])
plt.xlabel('Classifier')
plt.ylabel('Model Accuracy')
plt.title('Classifier Performance Comparison')
plt.show()

rfe_features = source_df[['Number of ratings', 'Dangerous permissions count', 'Safe permissions count']]
rfe_labels = source_df['Rating']

rfe_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
rfe_features_imputed = rfe_imputer.fit_transform(rfe_features)
rfe_labels_imputed = rfe_imputer.fit_transform(rfe_labels.values.reshape(-1, 1)).ravel()

rfe_linear_model = LinearRegression()
rfe_model = RFE(rfe_linear_model, n_features_to_select=2)
rfe_model.fit(rfe_features_imputed, rfe_labels_imputed)
print('RFE Selected Features:', rfe_features.columns[rfe_model.support_])

rating_class_features = source_df[['Dangerous permissions count', 'Safe permissions count']]
rating_class_labels = source_df['Rating'].apply(lambda val: 1 if val >= 4.0 else 0)

X_train_rating_cls, X_test_rating_cls, y_train_rating_cls, y_test_rating_cls = train_test_split(rfe_features_imputed, rating_class_labels, test_size=0.2, random_state=42)

lr_rating_cls = LogisticRegression().fit(X_train_rating_cls, y_train_rating_cls)
lr_rating_cls_preds = lr_rating_cls.predict(X_test_rating_cls)
print('LR Accuracy for Rating Class:', accuracy_score(y_test_rating_cls, lr_rating_cls_preds))

dt_rating_cls = DecisionTreeClassifier().fit(X_train_rating_cls, y_train_rating_cls)
dt_rating_cls_preds = dt_rating_cls.predict(X_test_rating_cls)
print('DT Accuracy for Rating Class:', accuracy_score(y_test_rating_cls, dt_rating_cls_preds))

rf_rating_cls = RandomForestClassifier().fit(X_train_rating_cls, y_train_rating_cls)
rf_rating_cls_preds = rf_rating_cls.predict(X_test_rating_cls)
print('RF Accuracy for Rating Class:', accuracy_score(y_test_rating_cls, rf_rating_cls_preds))

X_train_rating_reg, X_test_rating_reg, y_train_rating_reg, y_test_rating_reg = train_test_split(rfe_features_imputed, rfe_labels, test_size=0.2, random_state=42)

lr_rating_reg = LinearRegression().fit(X_train_rating_reg, y_train_rating_reg)
lr_rating_reg_preds = lr_rating_reg.predict(X_test_rating_reg)
print('Linear Regression MSE for Rating:', mean_squared_error(y_test_rating_reg, lr_rating_reg_preds))

dt_rating_reg = DecisionTreeRegressor().fit(X_train_rating_reg, y_train_rating_reg)
dt_rating_reg_preds = dt_rating_reg.predict(X_test_rating_reg)
print('Decision Tree MSE for Rating:', mean_squared_error(y_test_rating_reg, dt_rating_reg_preds))

rf_rating_reg = RandomForestRegressor().fit(X_train_rating_reg, y_train_rating_reg)
rf_rating_reg_preds = rf_rating_reg.predict(X_test_rating_reg)
print('Random Forest MSE for Rating:', mean_squared_error(y_test_rating_reg, rf_rating_reg_preds))

clustering_features = source_df[['Number of ratings', 'Dangerous permissions count', 'Safe permissions count']]
clustering_scaler = StandardScaler()
clustering_features_scaled = clustering_scaler.fit_transform(rfe_features_imputed)

kmeans_clusterer = KMeans(n_clusters=4, random_state=42, n_init=10).fit(clustering_features_scaled)
found_labels = kmeans_clusterer.labels_
source_df['Data_Cluster'] = found_labels

plt.figure(figsize=(10, 6))
plt.scatter(source_df['Number of ratings'], source_df['Dangerous permissions count'], c=source_df['Data_Cluster'])
plt.xlabel('Total Ratings')
plt.ylabel('High-Risk Permission Count')
plt.title('K-means Clustering of Apps')
plt.show()

pca_scaler = StandardScaler()
pca_features_scaled = pca_scaler.fit_transform(rfe_features_imputed)
pca_transformer = PCA(n_components=2).fit(pca_features_scaled)
pca_features = pca_transformer.transform(pca_features_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=source_df['Rating'], cmap='plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Application Features')
plt.show()

anomaly_detector = IsolationForest(contamination=0.08, random_state=42).fit(rfe_features)
detected_anomalies = anomaly_detector.predict(rfe_features)
source_df['Is_Anomaly'] = detected_anomalies
print('Total Anomalies Found:', len(source_df[source_df['Is_Anomaly'] == -1]))

sampling_feature_set = attributes_df.iloc[1:, :-1]
sampling_label_set = attributes_df.iloc[1:, -1].astype(int)

over_sampler_model = RandomOverSampler(sampling_strategy=0.85)
X_data_over, y_data_over = over_sampler_model.fit_resample(sampling_feature_set, sampling_label_set)
create_pie_chart(y_data_over, BALANCED_OVER)

X_train_over, X_test_over, y_train_over, y_test_over = train_test_split(X_data_over, y_data_over, test_size=0.2, random_state=42)
execute_logistic_regression(X_train_over, X_test_over, y_train_over, y_test_over)
execute_gaussian_nb(X_train_over, X_test_over, y_train_over, y_test_over)
dt_over_model = execute_decision_tree(X_train_over, X_test_over, y_train_over, y_test_over)
save_tree_visualization(dt_over_model, attributes_df.columns[:-1], BALANCED_OVER)

under_sampler_model = RandomUnderSampler(sampling_strategy=0.85)
X_data_under, y_data_under = under_sampler_model.fit_resample(sampling_feature_set, sampling_label_set)
create_pie_chart(y_data_under, BALANCED_UNDER)

X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(X_data_under, y_data_under, test_size=0.2, random_state=42)
execute_logistic_regression(X_train_under, X_test_under, y_train_under, y_test_under)
execute_gaussian_nb(X_train_under, X_test_under, y_train_under, y_test_under)
dt_under_model = execute_decision_tree(X_train_under, X_test_under, y_train_under, y_test_under)
save_tree_visualization(dt_under_model, attributes_df.columns[:-1], BALANCED_UNDER)

pca_main_cols = ['Rating', 'Number of ratings', 'Dangerous permissions count', 'Safe permissions count']
pca_main_imputer = SimpleImputer(strategy='median')
source_df[pca_main_cols] = pca_main_imputer.fit_transform(source_df[pca_main_cols])

pca_main_scaler = StandardScaler()
pca_main_data_scaled = pca_main_scaler.fit_transform(source_df[pca_main_cols])

pca_main_transformer = PCA(n_components=2).fit(pca_main_data_scaled)
pca_main_data = pca_main_transformer.transform(pca_main_data_scaled)

pca_main_df = pd.DataFrame(pca_main_data, columns=['PC1', 'PC2'])
pca_main_df['Rating'] = source_df['Rating']

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Rating', data=pca_main_df, palette='magma')
plt.title('Main PCA Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

full_pca = PCA(n_components=120)
full_principal_components = full_pca.fit_transform(attributes_df)
full_pca_variance_ratios = full_pca.explained_variance_ratio_ * 100

def create_pca_variance_plot(ratios, num_components):
    plt.figure(figsize=(12, 7))
    plt.xlabel('Principal Component Index')
    plt.ylabel('Explained Variance (%)')
    plt.xticks(range(0, num_components, 5), rotation=90)
    plt.title('PCA Explained Variance per Component')
    plt.tight_layout()
    plt.bar(range(num_components), ratios, color=sns.color_palette("rocket"))
    plt.savefig(FIGURES_DIR + 'PCA_Variance_Explained.jpg', bbox_inches='tight')
    plt.show()

create_pca_variance_plot(full_pca_variance_ratios, full_pca.n_components_)

reconstructed_df = pd.DataFrame(full_pca.inverse_transform(full_principal_components))
reconstructed_df[reconstructed_df <= 0] = 0
reconstructed_df[reconstructed_df != 0] = 1
reconstructed_df = reconstructed_df.astype(int)
reconstructed_df.index = source_df['App']

app_target_labels = source_df['Class']
app_target_labels.index = source_df['App']

pca_feature_df = pd.concat([reconstructed_df.iloc[:, :15], app_target_labels], axis=1)

X_features_pca = pca_feature_df.iloc[:, :-1]
y_labels_pca = pca_feature_df.iloc[:, -1]

X_train_pca_final, X_test_pca_final, y_train_pca_final, y_test_pca_final = train_test_split(X_features_pca, y_labels_pca, test_size=0.2, random_state=42)

execute_logistic_regression(X_train_pca_final, X_test_pca_final, y_train_pca_final, y_test_pca_final)
execute_gaussian_nb(X_train_pca_final, X_test_pca_final, y_train_pca_final, y_test_pca_final)
dt_pca_model = execute_decision_tree(X_train_pca_final, X_test_pca_final, y_train_pca_final, y_test_pca_final)
save_tree_visualization(dt_pca_model, [str(col) for col in pca_feature_df.columns[:-1]], RAW, 'pca_decision_tree')

def execute_random_forest(train_x, test_x, train_y, test_y):
    the_model = RandomForestClassifier(n_estimators=250, n_jobs=-1).fit(train_x, train_y)
    print_model_metrics(the_model, train_x, test_x, train_y, test_y)
    return the_model

execute_random_forest(X_train_pca_final, X_test_pca_final, y_train_pca_final, y_test_pca_final)

def execute_xgboost(train_x, test_x, train_y, test_y, f_name='xgboost_classifier.pkl'):
    the_model = GridSearchCV(estimator=xgb.XGBClassifier(scale_pos_weight=0.45, n_jobs=-1), param_grid={'max_depth': [3, 5]}, cv=5).fit(train_x, train_y)
    print_model_metrics(the_model, train_x, test_x, train_y, test_y)
    joblib.dump(the_model, f_name)
    return the_model

X_train_classifier, X_test_classifier, y_train_classifier, y_test_classifier = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
execute_xgboost(X_train_classifier, X_test_classifier, y_train_classifier, y_test_classifier)
execute_xgboost(X_train_pca_final, X_test_pca_final, y_train_pca_final, y_test_pca_final, f_name='xgboost_pca_classifier.pkl')

def execute_svm(train_x, test_x, train_y, test_y):
    the_model = svm.SVC(gamma='scale').fit(train_x, train_y)
    print_model_metrics(the_model, train_x, test_x, train_y, test_y)
    return the_model

execute_svm(X_train_pca_final, X_test_pca_final, y_train_pca_final, y_test_pca_final)

def execute_mlp(train_x, test_x, train_y, test_y):
    the_model = MLPClassifier(random_state=42, max_iter=350).fit(train_x, train_y)
    print_model_metrics(the_model, train_x, test_x, train_y, test_y)
    return the_model

execute_mlp(X_train_pca_final, X_test_pca_final, y_train_pca_final, y_test_pca_final)

final_model = joblib.load('xgboost_pca_classifier.pkl')
final_predictions = final_model.predict(X_test_pca_final)