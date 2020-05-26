import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, precision_recall_curve, auc,make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler,SMOTE
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import time
import sys
import ast


def aupr_scorer(test_y,y_pred):
	'''
	Function to calculate AUPR for given labels and predictions
	'''
	precision, recall, thresholds = precision_recall_curve(test_y,y_pred)
	aupr = auc(recall,precision)
	return aupr


def evaluateSampling(train_x, train_y, test_x, test_y, fractions,parameters,classifier,sampler):
	'''
	Function to evaluate how a particular sampling method affects the performance of a classifier on a holdout set
	The evaluation metrics considered here are AUPR, AUC and NLL

	args:
	train_x (pandas dataframe): training data insatnces without the labels
	train_y (numpy array): training data labels
	testing_x (pandas dataframe): testing data insatnces without the labels
	testing_y (numpy array): testing data labels
	fractions (list): different fractions to oversample or undersample the training set to
	parameters (dictionary): parameters for cross validation during grid search
	classifier (sklearn model): model
	sampler (imblearn sampler): sampling technique
	'''


	aupr_cvscorer = make_scorer(aupr_scorer)
	n_bootstrap = 25
	plt.figure(figsize=(12,20))
	plt.subplot(211)
	ax1 = plt.subplot(3, 1, 1)
	#ax1.set_xscale('log')
	ax1.set_xlabel('Degree of balance')
	ax1.set_ylabel('NLL')
	ax2 = plt.subplot(3, 1, 2)
	#ax3.set_xscale('log')
	ax2.set_xlabel('Degree of balance')
	ax2.set_ylabel('AUPR')
	ax3 = plt.subplot(3, 1, 3)
	#ax2.set_xscale('log')
	ax3.set_xlabel('Degree of balance')
	ax3.set_ylabel('AUC')
	

	mean_auc_us = []
	std_auc_us = []
	mean_aupr_us = []
	std_aupr_us = []
	std_nll_us = []
	mean_nll_us = []


	for f in tqdm(fractions):
	    aucs=[]
	    #precisions=[]
	    log_losses=[]
	    auprs = []
	    for r in tqdm(range(n_bootstrap)):
	        rs = sampler(sampling_strategy=f,random_state=r)
	        train_x_resampled, train_y_resampled = rs.fit_resample(train_x, train_y)

	        model = classifier['model']
	        cv_estimator = GridSearchCV(model, parameters, cv=3, scoring=aupr_cvscorer).fit(train_x_resampled, train_y_resampled)
	        clf = cv_estimator.best_estimator_.fit(train_x_resampled,train_y_resampled)
	        y_pred = clf.predict_proba(test_x)[:,1]

	        auprs.append(aupr_scorer(test_y,y_pred))
	        aucs.append(roc_auc_score(test_y,y_pred))
	        log_losses.append(log_loss(test_y,y_pred))
	   
	    mean_aupr_us.append(np.mean(auprs))
	    std_aupr_us.append(np.std(auprs))
	    mean_auc_us.append(np.mean(aucs))
	    std_auc_us.append(np.std(aucs))
	    mean_nll_us.append(np.mean(log_losses))
	    std_nll_us.append(np.std(log_losses))
	 
	    
	ax1.errorbar(fractions, mean_nll_us, std_nll_us)
	ax2.errorbar(fractions, mean_aupr_us, std_aupr_us)
	ax3.errorbar(fractions, mean_auc_us, std_auc_us)

	plt.show()

if __name__=='__main__':
	args = sys.argv[1:]
	train_x = pd.read_csv(args[0])
	test_x = pd.read_csv(args[2])
	train_y = pd.read_csv(args[1]).values
	test_y = pd.read_csv(args[3]).values

	print('Train and test dataframes created')

	models = {'logistic regression': {'model': LogisticRegression(solver='liblinear', max_iter=1000), 'parameters': {'C':[10**k for k in range(-3,4)],'penalty':['l2','l1']}},
			  'random forest': {'model': RandomForestClassifier(criterion='entropy', random_state=0), 'parameters': {'n_estimators':[30,50,100],'max_leaf_nodes':[3,7,10,15]}},
			  'gradient boosting classifier': {'model': GradientBoostingClassifier(random_state=0), 'parameters': {'n_estimators':[30,50,100],'max_leaf_nodes':[3,5,8,10]}}
			  }
	sampling_techniques = {'undersample':RandomUnderSampler,
						   'oversampler':RandomOverSampler,
						   'smote':SMOTE}
	classifier = models[args[4]]
	parameters = models[args[4]]['parameters']
	sampler = sampling_techniques[args[5]]
	fractions = [float(n) for n in ast.literal_eval(args[6])]
	print('Classifier: {}'.format(classifier))
	print('Parameters for CV: {}'.format(parameters))
	#print('Sampler: {}'.format(sampler))
	print('Fractions: {}'.format(fractions))
	start = time.time()
	evaluateSampling(train_x, train_y, test_x, test_y, fractions, parameters, classifier, sampler)
	print('Completion time: {} ms'.format(time.time()-start))

