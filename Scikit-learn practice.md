```python
# Get data ready
import pandas as pd
import numpy as np

heart_disease = pd.read_csv("heart-disease.csv")
```


```python
heart_disease
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>1</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>1</td>
      <td>1</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>1</td>
      <td>178</td>
      <td>0</td>
      <td>0.8</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>354</td>
      <td>0</td>
      <td>1</td>
      <td>163</td>
      <td>1</td>
      <td>0.6</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>298</th>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>140</td>
      <td>241</td>
      <td>0</td>
      <td>1</td>
      <td>123</td>
      <td>1</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>299</th>
      <td>45</td>
      <td>1</td>
      <td>3</td>
      <td>110</td>
      <td>264</td>
      <td>0</td>
      <td>1</td>
      <td>132</td>
      <td>0</td>
      <td>1.2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>300</th>
      <td>68</td>
      <td>1</td>
      <td>0</td>
      <td>144</td>
      <td>193</td>
      <td>1</td>
      <td>1</td>
      <td>141</td>
      <td>0</td>
      <td>3.4</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>301</th>
      <td>57</td>
      <td>1</td>
      <td>0</td>
      <td>130</td>
      <td>131</td>
      <td>0</td>
      <td>1</td>
      <td>115</td>
      <td>1</td>
      <td>1.2</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>302</th>
      <td>57</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>236</td>
      <td>0</td>
      <td>0</td>
      <td>174</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>303 rows × 14 columns</p>
</div>




```python
#predict the target
#Create X (features Matrix)
X = heart_disease.drop("target", axis=1)

#Create Y (labels)
Y = heart_disease["target"]
```


```python
#Choose the right model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators= 100)
clf.get_params()
```




    {'bootstrap': True,
     'ccp_alpha': 0.0,
     'class_weight': None,
     'criterion': 'gini',
     'max_depth': None,
     'max_features': 'auto',
     'max_leaf_nodes': None,
     'max_samples': None,
     'min_impurity_decrease': 0.0,
     'min_impurity_split': None,
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'min_weight_fraction_leaf': 0.0,
     'n_estimators': 100,
     'n_jobs': None,
     'oob_score': False,
     'random_state': None,
     'verbose': 0,
     'warm_start': False}




```python
#Fit the model to the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split (X, Y, test_size=0.2)
clf.fit(X_train, y_train);
```


```python
#make a prediction
y_label = clf.predict(np.array([0,2,3,4]))

```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-34-c9c706b2bc39> in <module>
          1 #make a prediction
    ----> 2 y_label = clf.predict(np.array([0,2,3,4]))
          3 y_preds = clf.predict(X_test)
          4 y_preds
    

    ~\anaconda3\lib\site-packages\sklearn\ensemble\_forest.py in predict(self, X)
        610             The predicted classes.
        611         """
    --> 612         proba = self.predict_proba(X)
        613 
        614         if self.n_outputs_ == 1:
    

    ~\anaconda3\lib\site-packages\sklearn\ensemble\_forest.py in predict_proba(self, X)
        654         check_is_fitted(self)
        655         # Check data
    --> 656         X = self._validate_X_predict(X)
        657 
        658         # Assign chunk of trees to jobs
    

    ~\anaconda3\lib\site-packages\sklearn\ensemble\_forest.py in _validate_X_predict(self, X)
        410         check_is_fitted(self)
        411 
    --> 412         return self.estimators_[0]._validate_X_predict(X, check_input=True)
        413 
        414     @property
    

    ~\anaconda3\lib\site-packages\sklearn\tree\_classes.py in _validate_X_predict(self, X, check_input)
        378         """Validate X whenever one tries to predict, apply, predict_proba"""
        379         if check_input:
    --> 380             X = check_array(X, dtype=DTYPE, accept_sparse="csr")
        381             if issparse(X) and (X.indices.dtype != np.intc or
        382                                 X.indptr.dtype != np.intc):
    

    ~\anaconda3\lib\site-packages\sklearn\utils\validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, warn_on_dtype, estimator)
        554                     "Reshape your data either using array.reshape(-1, 1) if "
        555                     "your data has a single feature or array.reshape(1, -1) "
    --> 556                     "if it contains a single sample.".format(array))
        557 
        558         # in the future np.flexible dtypes will be handled like object dtypes
    

    ValueError: Expected 2D array, got 1D array instead:
    array=[0. 2. 3. 4.].
    Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.



```python
y_preds = clf.predict(X_test)

```


```python
y_preds
```




    array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1,
           1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1,
           0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1], dtype=int64)




```python
#Evaluate the model
clf.score(X_train, y_train)
```




    1.0




```python
clf.score(X_test, y_test) #Scores this way because the model never seen the data either the label
```




    0.8688524590163934




```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test, y_preds))
```

                  precision    recall  f1-score   support
    
               0       0.89      0.83      0.86        29
               1       0.85      0.91      0.88        32
    
        accuracy                           0.87        61
       macro avg       0.87      0.87      0.87        61
    weighted avg       0.87      0.87      0.87        61
    
    


```python
confusion_matrix(y_test, y_preds)
```




    array([[24,  5],
           [ 3, 29]], dtype=int64)




```python
accuracy_score(y_test, y_preds)
```




    0.8688524590163934




```python
#Improve model
#Try different amount of n_estimators
np.random.seed(42)

for i in  range(10, 100, 10):
    print(f"Training model with {i} estimators...")
    clf = RandomForestClassifier(n_estimators=i).fit(X_train, y_train)
    print(f"model accuracy on test set: {clf.score(X_test, y_test) *100:.2f}%")
```

    Training model with 10 estimators...
    model accuracy on test set: 81.97%
    Training model with 20 estimators...
    model accuracy on test set: 85.25%
    Training model with 30 estimators...
    model accuracy on test set: 85.25%
    Training model with 40 estimators...
    model accuracy on test set: 85.25%
    Training model with 50 estimators...
    model accuracy on test set: 85.25%
    Training model with 60 estimators...
    model accuracy on test set: 85.25%
    Training model with 70 estimators...
    model accuracy on test set: 86.89%
    Training model with 80 estimators...
    model accuracy on test set: 83.61%
    Training model with 90 estimators...
    model accuracy on test set: 85.25%
    


```python
#Best traingin model accuracy with 70 estimators
```


```python

```
