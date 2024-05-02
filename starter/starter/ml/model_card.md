# Model Card


## Model Details
Random fores classifer using sklearn library that infiers salary based on all columns passed on the X_train dataframe.
hiper paramenters passed in to model are the following:
param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

## Intended Use
The inteded use is to infer salary from a api that receives all parameters requeried.
## Training Data
census.csv file 

## Evaluation Data
20% of data in the census.csv reserved for training
## Metrics
our model was evaluated on presicion recall and fbeta and got the following results accordingly:
presicion: 0.7791898332009531 
recall: 0.6244430299172502 
fbeta: 0.6932862190812721
## Ethical Considerations
As always a model cannot be 100% certain about a person's salary and should be trusted until certain point. it is by no meaning a an accurate representaion of a reality for a 100% of data points.
## Caveats and Recommendations
other models should be consider as they might give better results.
