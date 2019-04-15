import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import graphviz
import pydot

###### STEP: REEAD IN DATA ######

def read_file(filename):
	'''
	Reads in csv file and converts to pandas dataframe
	'''
	df = pd.read_csv(filename)
	return df

###### STEP: DISPLAY SUMMARY STATS #######

def calc_summary_stats(filename):
	'''
	Calculates mean, median, standard deviation, max and min values for all columns 
	(note: this may be non-sensical for categorical variables)
	'''
	df = read_file(filename)
	
	summary_df = pd.DataFrame(df.mean())
	summary_df = summary_df.rename(columns={0:'mean'})
	summary_df['std_dev'] = df.std()
	summary_df['median'] = df.median()
	summary_df['max_val'] = df.max()
	summary_df['min_val'] = df.min()

	return summary_df

def generate_summary_plot(filename, dep_var, predictor):
	'''
	Generates simple scatterplot for dependent variable and a given predictor
		to look at correlation / spot any outliers
	'''
	df = read_file(filename)
	df.plot.scatter(x=predictor, y=dep_var)

###### STEP: PRE-PROCESS DATA ######

def fill_missing(filename):
	'''
	fill in missing values with the median value of column
	'''
	df = read_file(filename)
	df.fillna(df.median(), inplace=True)
	return df

###### GENERATE FEATURES/PREDICTORS ######

def create_binned_col(df, col, bins, include_lowest=True):
	'''
	Takes a continuous variable and creates bins. The labels are simply ints,
	as this is what sklearn decision tree requires.
	'''
	labels = list(range(len(bins)-1))
	df[col] = pd.cut(df[col], bins, labels=labels, include_lowest=include_lowest)
	return df

def create_binary_col(df, col, criteria_dict):
	'''
	Maps values of a df column to 1 or 0 as outlined in the criteria_dict, taken
	as an input to the function
	'''
	df[col] = df[col].map(criteria_dict)
	return df

###### BUILD CLASSIFIER ######

def create_decision_tree(df, dep_var, max_depth, min_samples_split, predictors=None):
	'''
	Create a decision tree using sklearn. Requires pandas dataframe, list of
	predictors to use as input. If no predictors are input, it defaults to using
	all potential predictors

	Creates separate training, testing data

	Returns predicted y-values and y-testing values; also creates dot file to
	visualize tree
	'''
	#Separate dependent variable and predictors
	Y = df[dep_var]

	if not predictors:
		X = df.drop(dep_var, axis=1)

	else:
		X = df[predictors]

	#Separate training and test data
	X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
	
	#Create model
	model = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
	model = model.fit(X_train, y_train)
	y_predict = model.predict(X_test)

	tree.export_graphviz(model, out_file='tree.dot', feature_names=X.columns)

	return (y_test, y_predict)

def evaluate_decision_tree(y_test, y_predict):
	'''
	Takes output of create decision tree to evaluate its accuracy.
	'''
	return accuracy_score(y_test, y_predict)


