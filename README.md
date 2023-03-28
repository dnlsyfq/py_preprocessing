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
