# Preprocessing

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

## Normalization
* high variance data do normalization
```
np.log(df[col])
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

```

## Encoding Numerical 

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
