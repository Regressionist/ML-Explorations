This folder contains some experiments I did on class imbalance. The file sampling.py can be used for evaluating how changing the distribution of labels in the training set by oversampling or undersampling affects the performance of 3 classifiers (logisitc regression, random forest, and gradient boostong classifier) on the test set. The evaluation metrics used are NLL loss, AUPR, and AUC. 

To run the file, use 
python sampling.py train_x train_y test_x test_y model sampling_technique fractions 

where train_x, train_y, test_x, test_y are the locations of these 4 csv files, model is one of ['logisitc regression', 'random forest', 'gradient boostong classifier'], sampling technique is one of ['undersample', 'oversample', 'smote'] and fractions are the degree of balances that you want to have in your training set. 

Example of the command is:
python sampling.py 'data/creditcard/train_x.csv' 'data/creditcard/train_y.csv' 'data/creditcard/test_x.csv' 'data/creditcard/test_y.csv' 'logistic regression' 'undersample' '[1,0.2,0.02,0.002]'