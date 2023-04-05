# Preprocessing

## missing value
```
* listwise deletion (delete entire records when 1 column has na)
df.dropna(how='any')

* drop missing values in specific col
df.dropnna(subset=['VersionControl']

* drop missing values in all col
df.dropna(how='any',axis=1)

df.drop(columns=[col])
```

## fill
```
df[col].fillna(val,inplace=True)

```
One approach to finding these values is to force the column to the data type desired using pd.to_numeric()


## Split
```
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
```

## Class Imbalance
```
x_train, x_test, y_train, y_test = train_test_split(X,y,stratify=y,random_state=42)
```

## Standardization
* transform continuous numerical data to appear normally distributed
* non normal training data introduce bias 

when to standardize  
  * dataset with high variance  
  * features on different scales
  * use distance metric in linear space
    * knn
    * kmeans
    * linear regression


```
# wine dataset, One of the columns, Proline, has an extremely high variance compared to the other columns.
wine.describe().loc['std']
```

```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df[['Age']])
df['standardized_col'] = scaler.transform(df[['Age']])
```

```
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Instantiate StandardScaler
SS_scaler = StandardScaler()

# Fit SS_scaler to the data
SS_scaler.fit(so_numeric_df[['Age']])

# Transform the data using the fitted scaler
so_numeric_df['Age_SS'] = SS_scaler.transform(so_numeric_df[['Age']])

# Compare the origional and transformed column
print(so_numeric_df[['Age_SS', 'Age']].head())
```

## Normalization
* high variance data do normalization
* make highly skewed to less skewed
```
np.log(df[col])
```
```
from sklearn.preprocessing import PowerTransformer
log = PowerTransformer()
log.fit(df[['ConvertedSalary']])
df['log_ConvertedSalary'] = log.transform(df[['ConvertedSalary']])
```
```
# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Instantiate MinMaxScaler
MM_scaler = MinMaxScaler()

# Fit MM_scaler to the data
MM_scaler.fit(so_numeric_df[['Age']])

# Transform the data using the fitted scaler
so_numeric_df['Age_MM'] = MM_scaler.transform(so_numeric_df[['Age']])

# Compare the origional and transformed column
print(so_numeric_df[['Age_MM', 'Age']].head())
```



```
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df[['Age']])
df['normalized_age'] = scaler.transform(df[['Age']])
```

## Feature Scaling
* all columns in different scale then need to scaling
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
```

## Encoding categorical variables



### Encoding binary variables
* Yes, No
```
df[col] = df[col].apply(lambda val: 1 if val == 'Yes' else 0)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df[col] = le.fit_transform(df[col])

```

### Encoding more than 2 values
* one hot encoding
```
pd.get_dummies(df[col)
pd.get_dummies(df,columns=[col],prefix='C')
```
* dummy encoding
```
pd.get_dummies(df,columns=[col],drop_first=True,prefix='C')
```

avoid too many columns
```
counts = df[col].value_counts()

mask = df[col].isin(counst[counts<5].index)
df[col][mask] = 'other
```

* masking
```
# Create a series out of the Country column
countries = so_survey_df['Country']

# Get the counts of each category
country_counts = countries.value_counts()

# Create a mask for only categories that occur less than 10 times
mask = countries.isin(country_counts[country_counts < 10].index)
print(mask)

# Label all other categories as Other
countries[mask] = 'Other'

# Print the updated category counts
print(countries.value_counts())
```

## Encoding Numerical 

* binning numerical using pd.cut
For example on some occasions, you might not care about the magnitude of a value but only care about its direction, or if it exists at all. In these situations, you will want to binarize a column
```
df['Binned_Group'] = pd.cut(df['Number_of_Violations'],bins=[-np.inf,0,2,np.inf],labels=[1,2,3])
```

```
# Import numpy
import numpy as np

# Specify the boundaries of the bins
bins = [-np.inf, 10000, 50000, 100000, 150000, np.inf]

# Bin labels
labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']

# Bin the continuous variable ConvertedSalary using these boundaries
so_survey_df['boundary_binned'] = pd.cut(
    so_survey_df['ConvertedSalary'],bins=bins,labels=labels)

# Print the first 5 rows of the boundary_binned column
print(so_survey_df[['boundary_binned', 'ConvertedSalary']].head())
```



```
* Scaled

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neigbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,random_state=42)
knn = KNeighborsClassifier()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn.fit(X_train_scaled,y_train)
knn.score(X_test_scaled,y_test)

* Non Scaled

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
```

## Removing outliers

* 95 quantile
```
q_cutoff = df[col].quantile(0.95)
mask = df[col] < q_cutoff
trimmed_df = df[mask]
```

* 3 standard deviation from the mean
```
mean = df[col].mean()
std = df[col].std()
cut_off = std * 3

lower, upper = mean - cut_off , mean + cut_off

new_df = df[ (df[col] < upper) & (df[col] < lower) ]




```

### Train and Test
```
scaler = StandardScaler()
scaler.fit(train[[col]])
train[colscaled] = scaler.transform(train[[col]])

test[colscaled] = scaler.transform(test[col]]
```
```
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Apply a standard scaler to the data
SS_scaler = StandardScaler()

# Fit the standard scaler to the data
SS_scaler.fit(so_train_numeric[['Age']])

# Transform the test data using the fitted scaler
so_test_numeric['Age_ss'] = SS_scaler.transform(so_test_numeric[['Age']])
print(so_test_numeric[['Age', 'Age_ss']].head())
```



same goes with removing outliers using traning lower and upper 
```
train_std = so_train_numeric['ConvertedSalary'].std()
train_mean = so_train_numeric['ConvertedSalary'].mean()

cut_off = train_std * 3
train_lower, train_upper = train_mean - cut_off, train_mean + cut_off

# Trim the test DataFrame
trimmed_df = so_test_numeric[(so_test_numeric['ConvertedSalary'] < train_upper) \
                             & (so_test_numeric['ConvertedSalary'] > train_lower)]
```



## Dimensionality Reduction
```
from sklearn.decomposition import PCA
pca = PCA()
print(pca.explained_variance_ratio)
df_pca = pca.fit_transform(df)

```

```
# Instantiate a PCA object
pca = PCA()

# Define the features and labels from the wine dataset
X = wine.drop("Type", axis=1)
y = wine["Type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Apply PCA to the wine dataset X vector
pca_X_train = pca.fit_transform(X_train)
pca_X_test = pca.transform(X_test)

# Look at the percentage of variance explained by the different components
print(pca.explained_variance_ratio_)

# Fit knn to the training data
knn.fit(pca_X_train,y_train)

# Score knn on the test data and print it out
knn.score(pca_X_test,y_test)
```
