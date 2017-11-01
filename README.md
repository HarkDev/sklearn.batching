# sklearn.onsteroids
A set of classes to implement everyday's Data Scientists development work, into a set of libraries that implement the SciKitLearn API. 
This allows for implementing common actions into the a scikit learn's Pipeline

Implements:
* Batching
* Feature Engineering techniques

# Who is this for?

Data Science is a great career. It;s really easy if you work at a company where they have the equipment needed to work with huge datasets.

But if you are an independent, freelancer, or creating a small startup, sometimes, big equipment is not possible, and you are out of the game.

I'm creating this set of classes that implement some ways to tackle this problem, and make the game a bti more balanced. 

# How does it work?

Basically, contains a set of classes that implement the SciKit Leran's API, that allow to apply batching on any batching on
* transforming
* scaling
* applying feature engineering
* training 
* predicting

# What is implemented?
The following are some of the classes that are implementes:

## Batching Standard Scaler
Allow to apply scaling on a batch with 2 lines of code:
```
# Batch standard scaler
std_scaler = BatchStandardScaler(columns = SCALE_COLUMNS, batch_size=100000)

# Fit and transform on batches
df_train = std_scaler.fit_transform(df_train)
```
```
 Partially fitting standard scaler...
[ ================================================== ] 100.00%
 Partially transforming...
[ =============================                      ] 59.32%
```
**Feel free to fork and pull new commits on your custom implementations of batching or any other techniques to use big databases**
