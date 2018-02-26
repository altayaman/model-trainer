import numpy as np
import pandas
import sys
import getpass
from optparse import OptionParser
import pickle
import time
from sqlalchemy import create_engine
from configparser import SafeConfigParser
from IPython.display import display, HTML
# Feature creator libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
# ML classifier libraries
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
# Aux libraries for ML classifiers
from scipy import stats
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score



## Read arguments
def get_args():
    help_text = """CME Node Recall Booster"""

    parser = OptionParser(usage=help_text)
    parser.add_option("-H", "--host",           dest="host",            default = "", help="the url for the DB", metavar="HOST_ADDRESS")
    parser.add_option("-d", "--db",             dest="database",        default = "",             help="the logical db name")
    parser.add_option("-u", "--username",       dest="username",        default = "",  help="the username for the DB", metavar="NAME")
    parser.add_option("-p", "--password",       dest="password",        help="the password for the DB", metavar="PASS")
    parser.add_option("-i", "--input_table",    dest="input_table")
    parser.add_option("-f", "--file",           dest="file")
    parser.add_option("--pkl", "--pickle",      dest="pickle",          help="the pickle file if data is cached")
    parser.add_option("--cf", "--config_file",  dest="config_file",     help="config_file that has all settings")

    (options, args) = parser.parse_args()

    return (options, args)

def check_if_file_exists(file_path):
    try:
        with open(file_path) as infile:
            pass
    except IOError:
        print('  ERROR: '+file_path+' file not found\n')
        sys.exit()

def get_db_engine_2(config_set_, username_, password_, host_, database_, port_):
    # for RedShift
    if(config_set_ == 'redshift'):
        url = ''.join(['postgresql://', username_, ":", password_, "@", host_, ':',port_, '/', database_])

    # for MsSQL
    elif(config_set_ == 'mssql'):
        url = ''.join(['mssql+pymssql://', username_, ":", password_, "@", host_, ':',port_, '/', database_])

    print(url)
    engine = create_engine(url)
    return engine

def get_db_engine(options):
    url = "".join(["postgresql://", options.username, ":", options.password, "@", options.host, ":5439/", options.database])
    engine = create_engine(url)
    return engine

def get_data_df_2(options_, args):
    parser = SafeConfigParser()

    if(options_.config_file):
        parser.read(options_.config_file)
        
        # Get train data source args
        if('train_data' in parser.options('training')):
            train_args_str = parser.get('training', 'train_data')
            config_set, config_element = train_args_str.split('+')
        elif('pickle' in parser.options('training')):
            config_set = parser.get('training', 'pickle')

            pkl_file = parser.get('training', 'pickle')
            check_if_file_exists(pkl_file)

            print('Reading train data from pickle ' + pkl_file + ' ...')
            df = pandas.read_pickle(pkl_file)

            print("Found " + str(len(df)) + " entries")
            return df
        else:
            print('One of the following options should be in training arg section:')
            print('train_data and pickle')
            print('or just pickle')
            sys.exit()


        if(config_set == 'config_file'):
            train_file_name = parser.get(config_set, config_element)
            check_if_file_exists(train_file_name)

            print('Reading train data from file ' + train_file_name + ' ...')
            df = pandas.read_csv(train_file_name)
            print(train_file_name)
            print(df.shape)

            #if parser.get('training', 'pickle'):
            if('pickle' in parser.options('training')):
                print(parser.get('training', 'pickle'))
                df.to_pickle(parser.get('training', 'pickle'))

        #elif(config_set == 'config_redshift'):
        elif(parser.get(config_set, 'db_type') == 'redshift'):
            # Get database engine
            db_type  = parser.get(config_set, 'db_type')
            username = parser.get(config_set, 'username')
            password = parser.get(config_set, 'password')
            host     = parser.get(config_set, 'host')
            database = parser.get(config_set, 'database')
            port     = parser.get(config_set, 'port')
            engine   = get_db_engine_2(db_type, username, password, host, database, port)
            print("\nEstablished connection with the database")

            # Fetch data from db
            start = time.time()
            #print("\nReading input data from database ...")
            train_data_table_name = parser.get(config_set, config_element)
            print("\nReading train data from RedShift table " + train_data_table_name + "...")
            df = pandas.read_sql_query('SELECT * FROM ' + train_data_table_name, engine)

            ## Get elapsed time
            end = time.time()
            print("Reading data from database took %g s" % (end - start))

            #if parser.get('training', 'pickle'):
            if('pickle' in parser.options('training')):
                print(parser.get('training', 'pickle'))
                df.to_pickle(parser.get('training', 'pickle'))

        #elif(config_set == 'config_mssql'):
        elif(parser.get(config_set, 'db_type') == 'mssql'):
            # Get database engine
            db_type  = parser.get(config_set, 'db_type')
            username = parser.get(config_set, 'username')
            password = parser.get(config_set, 'password')
            host     = parser.get(config_set, 'host')
            database = parser.get(config_set, 'database')
            port     = parser.get(config_set, 'port')
            engine   = get_db_engine_2(db_type, username, password, host, database, port)
            print("\nEstablished connection with the database")

            # Fetch data from db
            start = time.time()
            #print("\nReading input data from database ...")
            train_data_table_name = parser.get(config_set, config_element)
            print("\nReading train data from MsSQL table " + train_data_table_name + "...")
            df = pandas.read_sql_query('SELECT * FROM ' + train_data_table_name, engine)

            ## Get elapsed time
            end = time.time()
            print("Reading data from database took %g s" % (end - start))

            #if parser.get('training', 'pickle'):
            if('pickle' in parser.options('training')):
                print(parser.get('training', 'pickle'))
                df.to_pickle(parser.get('training', 'pickle'))

        elif parser.get('training', 'pickle'):
            pkl_file = parser.get('training', 'pickle')
            check_if_file_exists(pkl_file)
            df = pandas.read_pickle(pkl_file)

        else:
            print("Need to specify either input db table or pickle file or input file")
            sys.exit()

        print("Found " + str(len(df)) + " entries")
        return df
    else:
        print(' ERROR: --cf option is not passed')
        sys.exit()

def get_data_df(options, args):

    if options.file is not None:
        check_if_file_exists(options.file)
        df = pandas.read_csv(options.file)
        print(df.shape)
    elif options.input_table is not None:
        # Read db password and save in options
        password = getpass.getpass("\nPlease enter the password for the DB: ")
        options.password = password

        # Get database engine
        engine = get_db_engine(options)
        print("\nEstablished connection with the database")

        start = time.time()
        print("\nReading input data from database ...")
        df = pandas.read_sql_query('SELECT * FROM ' + options.input_table, engine)

        ## Get elapsed time
        end = time.time()
        print("Reading data from database took %g s" % (end - start))
        
        if options.pickle is not None:
            df.to_pickle(options.pickle)
    elif options.pickle is not None:
        df = pandas.read_pickle(options.pickle)
    else:
        print("Need to specify either input db table or pickle file or input file")
        sys.exit()
        
    print("Found " + str(len(df)) + " entries")
    
    return df

def get_positives_negatives(df, category):
    positives = df[(df.category_full_path_mod1 == category) & (df.type == 'True Positive')].loc[:,'description_mod1']
    negatives = df[((df.category_full_path_mod1 != category) & (df.type == 'True Positive')) | \
                   ((df.category_full_path_mod1 == category) & (df.type == 'False Positive'))
                  ].loc[:,'description_mod1']
    return (positives.drop_duplicates(), negatives.drop_duplicates())

def get_vectorized_data(positives, negatives, count_vect=None, tfidf_vect=None):
    if count_vect and tfidf_vect:
        X_train_counts = count_vect.fit_transform(pandas.concat([positives, negatives]))
        X_train_tfidf  = tfidf_vect.fit_transform(pandas.concat([positives, negatives]))
        X_train = scipy.sparse.hstack([X_train_counts, X_train_tfidf])
        print('      Both of vectorizers',X_train.shape)
    elif count_vect and tfidf_vect is None:
        X_train = count_vect.fit_transform(pandas.concat([positives, negatives]))
        print('      Countvect only',X_train.shape)
    elif count_vect is None and tfidf_vect:
        X_train  = tfidf_vect.fit_transform(pandas.concat([positives, negatives]))
        print('      TfIdf only',X_train.shape)
    else:
        print('      None of Vectorizers')

    Y = [1] * len(positives) + [0] * len(negatives)
    return (X_train, Y), count_vect

def get_trained_clf(df, category, X_train_counts, Y, count_vect, clf, clf_name):
    ## get validation score for given classifier
    print('  Cross validation for ' + clf_name)
    scores = cross_val_score(clf, X_train_counts, Y, scoring='recall', cv=5)
    params = None
    gs_clf = None

    # set clf into grid search
    if(isinstance(clf, tree.DecisionTreeClassifier)):
        print('  clf is tree.DecisionTreeClassifier ...')
        params = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
        gs_clf = GridSearchCV(estimator = clf, param_grid = params, cv=5, n_jobs = -1)
    elif(isinstance(clf, LogisticRegression)):
        print('  clf is Logistic Reg ...')
        params = {'C': [0.001, 0.01, 0.1, 1, 10, 15, 20, 30, 40, 100, 1000] }
        gs_clf = GridSearchCV(estimator = clf, param_grid = params, cv=5, n_jobs = -1)
        print('  clf is Random Forest ...')
    elif(isinstance(clf, RandomForestClassifier)):
        params = {"max_depth": [3, None],
                      "max_features": [1, 3, 10],
                      "min_samples_split": [1, 3, 10],
                      "min_samples_leaf": [1, 3, 10],
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]}
        gs_clf = GridSearchCV(estimator = clf, param_grid = params, cv=5, n_jobs = -1)
    elif(isinstance(clf, svm.SVC)):
        print('  clf is SVM ...')
        C_range = 10.0 ** np.arange(-4, 4)
        gamma_range = 10.0 ** np.arange(-4, 4)
        params = {'C': C_range.tolist(), 'gamma': gamma_range.tolist()}
        gs_clf = GridSearchCV(estimator = clf, param_grid = params, cv=5, n_jobs = -1)

    ## train classifier for recall and precision measurements
    gs_clf.fit(X_train_counts, Y)
    
    ## recall measurement
    false_negatives = df[((df.category_full_path_mod1 == category) & (df.type == 'False Negative'))].loc[:,'description_mod1']
    false_negatives = false_negatives.drop_duplicates()
    #X_test_counts = count_vect.transform(false_negatives)
    #Y_test = clf.predict(X_test_counts)

    ## precision measurement
    false_positives = df[((df.category_full_path_mod1 != category) & (df.type == 'False Negative'))].loc[:,'description_mod1']
    false_positives = false_positives.drop_duplicates()
    #X_test_counts2 = count_vect.transform(false_positives)
    #Y_test2 = clf.predict(X_test_counts2)

    ## Persist classifier and it's scores to dict
    results_dict = {}
    results_dict["Model name"] = clf_name
    results_dict["Cross Validation Score"] = np.mean(scores)
    #results_dict["Recall"] = np.sum(Y_test)*1.0/len(Y_test)
    #results_dict["Precision"] = 1 - np.sum(Y_test2)*1.0/len(Y_test2)
    results_dict["Model"] = gs_clf

    for param_name in sorted(params.keys()):
        results_dict[param_name] = gs_clf.best_params_[param_name]

    return results_dict

def get_trained_clf_2(df, category, X_train_counts, Y, count_vect, clf, clf_name):
    params = None
    gs_clf = None

    # set clf into grid search
    if(isinstance(clf, tree.DecisionTreeClassifier)):
        print('='*100)
        print('      Optimizing tree.DecisionTreeClassifier ...')
        params = {'criterion':['gini','entropy'],
                  'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
        gs_clf = RandomizedSearchCV(estimator = clf, param_distributions = params, cv=5, n_jobs = -1)
    elif(isinstance(clf, LogisticRegression)):
        print('='*100)
        print('      Optimizing Logistic Reg ...')
        params = {'C': [0.001, 0.01, 0.1, 1, 10, 15, 20, 30, 40, 100, 1000],
                  'penalty': ['l1','l2']}
        gs_clf = RandomizedSearchCV(estimator = clf, param_distributions = params, cv=5, n_jobs = -1)
    elif(isinstance(clf, RandomForestClassifier)):
        print('='*100)
        print('      Optimizing Random Forest ...')
        params = {"max_depth": [3, 5, None],
                  "max_features": [1, 2, 3, 4, 5, 7, 9],
                  "min_samples_split": [1, 2, 3, 4, 5, 7, 9],
                  "min_samples_leaf": [1, 2, 3, 4, 5, 7, 9],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
        gs_clf = RandomizedSearchCV(estimator = clf, param_distributions = params, cv=5, n_jobs = -1)
    elif(isinstance(clf, svm.SVC)):
        print('='*100)
        print('      Optimizing SVM ...')
        C_range = 10.0 ** np.arange(-4, 4)
        gamma_range = 10.0 ** np.arange(-4, 4)
        kernels = ['rbf','linear','poly','sigmoid']
        params = {'C': C_range.tolist(), 
                  'gamma': gamma_range.tolist(), 
                  'kernel': kernels}
        gs_clf = RandomizedSearchCV(estimator = clf, param_distributions = params, cv=5, n_jobs = -1)
    elif(isinstance(clf, MultinomialNB)):
        print('='*100)
        print('      Optimizing MultinomialNB ...')
        params = {'alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001], 
                  'fit_prior': [True, False]}
        gs_clf = RandomizedSearchCV(estimator = clf, param_distributions = params, cv=5, n_jobs = -1, n_iter = 10)
    elif(isinstance(clf, AdaBoostClassifier)):
        print('='*100)
        print('      Optimizing Ada Boost ...')
        params = {'learning_rate': stats.expon(scale=1.0), 
                  'n_estimators': stats.randint(low=20, high=100)}
        gs_clf = RandomizedSearchCV(estimator = clf, param_distributions = params, cv=5, n_jobs = -1, n_iter = 10)
    elif(isinstance(clf, KNeighborsClassifier)):
        print('='*100)
        print('      Optimizing KNN Neighbors ...')
        params = {'n_neighbors': [i for i in range(2, 10)], 
                  'weights': ['uniform', 'distance']}
        gs_clf = RandomizedSearchCV(estimator = clf, param_distributions = params, cv=5, n_jobs = -1, n_iter = 10)


    start = time.time()

    ## train classifier for recall and precision measurements
    gs_clf.fit(X_train_counts, Y)

    print("      Optimization process took %g s" % (time.time()- start))
    
    ## get validation score for given classifier
    print('      Cross validation for ' + clf_name)
    scores = cross_val_score(gs_clf, X_train_counts, Y, scoring='recall', cv=5)

    ## Print accuracy
    predictions = gs_clf.predict(X_train_counts)
    print('\n      Best score: ', np.mean(scores))
    print('      Prediction accuracy score: ', accuracy_score(Y, predictions))
    print('      Confusion matrix:')
    display(pandas.crosstab(pandas.Series(Y), predictions, rownames=['True'], colnames=['Predicted'], margins=True))
    print('\n')

    ## recall measurement
    #false_negatives = df[((df.category_full_path_mod1 == category) & (df.type == 'False Negative'))].loc[:,'description_mod1']
    #false_negatives = false_negatives.drop_duplicates()
    #X_test_counts = count_vect.transform(false_negatives)
    #Y_test = clf.predict(X_test_counts)

    ## precision measurement
    #false_positives = df[((df.category_full_path_mod1 != category) & (df.type == 'False Negative'))].loc[:,'description_mod1']
    #false_positives = false_positives.drop_duplicates()
    #X_test_counts2 = count_vect.transform(false_positives)
    #Y_test2 = clf.predict(X_test_counts2)

    ## Persist classifier and it's scores to dict
    results_dict = {}
    results_dict["Model name"] = clf_name
    #results_dict["Cross Validation Score"] = np.mean(scores)
    #results_dict["Best Score"] = gs_clf.best_score_
    results_dict["Best Score"] = np.mean(scores)
    #results_dict["Recall"] = np.sum(Y_test)*1.0/len(Y_test)
    #results_dict["Precision"] = 1 - np.sum(Y_test2)*1.0/len(Y_test2)
    results_dict["Model"] = gs_clf

    for param_name in sorted(params.keys()):
        results_dict[param_name] = gs_clf.best_params_[param_name]

    return results_dict

def get_best_local_model(clf_dict, df, category, X_train_counts, Y, count_vect):
    import copy
    
    models_list = []
    
    ## create models and add to list
    for clf_name, clf in clf_dict.items():
        #model_trained = get_trained_clf(df, category, X_train_counts, Y, count_vect, copy.deepcopy(clf), clf_name)
        model_trained = get_trained_clf_2(df, category, X_train_counts, Y, count_vect, copy.deepcopy(clf), clf_name)
        models_list.append(model_trained)
    
    ## Get model with highest validation score
    best_model_score_tuple = max(models_list, key = lambda model_score:model_score['Best Score'])
    
    return best_model_score_tuple

def compare_and_pick_best_model(best_local_model, best_global_model):    
    if not best_local_model:
        return best_global_model
    elif not best_global_model:
        return best_local_model
    elif(best_global_model['Best Score'] < best_local_model['Best Score']):
        return best_local_model
    else:
        return best_global_model

def get_best_model_2(data_df, category, clf_dict, ng_range = (1,1), verbous = False):
    # Result holder for final classifier
    best_final_clf = {}
    
    # Get (positives, negatives) for training set
    (positives, negatives) = get_positives_negatives(data_df, category[0])
    
    # Start running classifiers for different feature counts
    #for num_features_val in range(50,550,50):  # [50,100 ... 450,500]
    #    print('Num of features: ', num_features_val)
    
    for ng_range in [(1,1), (1,2), (2,2)]:
        print('='*100)
        print('Trial for n-gram range:',ng_range)

        # Create training set
        print('   Vectorizing training data ...')
        count_vect = CountVectorizer(min_df=1, ngram_range=ng_range, binary = True, stop_words="english")
        tfidf_vect = TfidfVectorizer(sublinear_tf=True, max_df=1.0, ngram_range=ng_range, stop_words='english')
        (X_train_counts, Y), count_vect_fitted = get_vectorized_data(positives, negatives, count_vect=count_vect, tfidf_vect=None)

        # Get best classifier for current iteration (i.e. features) as local
        print('   Testing classifiers ...')
        best_local_clf = get_best_local_model(clf_dict, data_df, category[0], X_train_counts, Y, count_vect)
        
        # Add additional meta-data
        #best_local_clf['Num of features']  = num_features_val
        best_local_clf['Count vectorizer'] = count_vect_fitted
        best_local_clf['Category name'] = category[0]
        best_local_clf['Category ID']   = category[1]
        best_local_clf['ngram range']  = ng_range
        
        
        if(verbous == True):
            print('Selected model for range ', ng_range)
            print_(best_local_clf, indent ='      ', print_all = True)            

        # Compare local classifier with previous, and get the best as final
        best_final_clf = compare_and_pick_best_model(best_final_clf, best_local_clf)
        
    
    return best_final_clf

def get_best_model(data_df, category, clf_dict, verbous = False):
    # Result holder for final classifier
    best_final_clf = {}
    
    # Get (positives, negatives) for training set
    (positives, negatives) = get_positives_negatives(data_df, category[0])
    
    # Start running classifiers for different feature counts
    for num_features_val in range(50,550,50):  # [50,100 ... 450,500]
        print('Num of features: ', num_features_val)
        
        # Create training set
        count_vect = CountVectorizer(max_features=num_features_val, min_df=1, ngram_range=(1, 1))
        (X_train_counts, Y), count_vect_ = get_vectorized_data(positives, negatives, count_vect)

        # Get best classifier for current iteration (i.e. features) as local
        best_local_clf = get_best_local_model(clf_dict, data_df, category[0], X_train_counts, Y, count_vect)
        
        # Add additional meta-data
        best_local_clf['Num of features']  = num_features_val
        best_local_clf['Count vectorizer'] = count_vect_
        best_local_clf['Category name'] = category[0]
        best_local_clf['Category ID']   = category[1]
        
        
        if(verbous == True):
            print_(best_local_clf, print_all = True)            

        # Compare local classifier with previous, and get the best as final
        best_final_clf = compare_and_pick_best_model(best_final_clf, best_local_clf)
        
    
    return best_final_clf
  
def export_model_file(best_final_clf_):
    print('Exporting model files ...')
    #for category in selected_models_by_category_.keys():
        # Create pickle file name for classifier
    model_file_name = 'category_' + str(best_final_clf_['Category ID']) + '_model.pkl'
    
    # Save model as pickle file
    with open(model_file_name, 'wb') as pickle_file:
        pickle.dump(best_final_clf_, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    print('Exported model file: ', model_file_name)

def export_model_files(selected_models_by_category_):
    print('Exporting model files ...')
    for category in selected_models_by_category_.keys():
        # Create pickle file name for classifier
        model_file_name = 'category_' + str(selected_models_by_category_[category]['Category ID']) + '_model.pkl'
        
        # Save model as pickle file
        with open(model_file_name, 'wb') as pickle_file:
            pickle.dump(selected_models_by_category_[category], pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        print('Exported model file: ', model_file_name)

def print_2(best_local_model, print_all = False):
    for k,v in best_local_model.items():
        if(not isinstance(v,CountVectorizer) and not isinstance(v,RandomizedSearchCV)):
            print(k,v)

def print_(best_local_model, indent = '', print_all = False):
    print(indent + 'Model name:       ', best_local_model['Model name'])
    #print(indent + 'Validation Score: ', best_local_model['Cross Validation Score'])
    print(indent + 'Best Score: ', best_local_model['Best Score'])
    #print(indent + 'Num of features:  ', best_local_model['Num of features'])
    print(indent + 'ngram range:  ', best_local_model['ngram range'])
    
    if(print_all == True):
        #print(indent + 'Precision:        ', best_local_model['Precision'])
        #print(indent + 'Recall:           ', best_local_model['Recall'])
        print(indent + 'Category name:    ', best_local_model['Category name'])
        print(indent + 'Category ID:      ', best_local_model['Category ID'])
    print()
    
def print_selected_models(selected_models_by_category, print_all = False):
    for category_tuple, model in selected_models_by_category.items():
        print(category_tuple)
        print_2(model, print_all = print_all)
        print('\n')


def main_1():
    start = time.time()
    ## Get input arguments
    (options, args) = get_args()

    ## Fetch input data to be classified
    #train_data_df = get_data_df(options, args)
    train_data_df = get_data_df_2(options, args)


    #sys.exit()
    #print(train_data_df.shape)
    #print(train_data_df.columns)
    
    # Result holder for all categories
    selected_models_by_category = {}

    # Get unique list of categories with ids
    categories = pandas.unique(train_data_df.loc[:,['category_full_path_mod1','category_id_mod1']].values)

    # Indicate classifiers to be tested
    clf_dict = {}
    #clf_dict['Decision Tree']       = tree.DecisionTreeClassifier()
    #clf_dict['Random Forest']       = RandomForestClassifier()
    clf_dict['Logistic Regression'] = LogisticRegression()
    #clf_dict['SVM']                 = svm.SVC(probability=True)
    #clf_dict['MultinomialNB']       = MultinomialNB()
    #clf_dict['AdaBoost']            = AdaBoostClassifier()
    #clf_dict['KNeighbors']          = KNeighborsClassifier()

    # Start iterating categories through classifiers
    for category in categories:    
        print('='*100,'\nRun models for following category:\n', category[0], '\n', '='*100)

        # Get best model with meta-info
        best_final_clf = get_best_model_2(train_data_df, category, clf_dict, verbous = True)
        
        # Print best model scores for current category
        print('SELECTED MODEL FOR  ' + category[0] + ':')
        print_2(best_final_clf, print_all = True)
        print('\n',best_final_clf,'\n')

        # Persist final classifier for current category to dict
        selected_models_by_category[category] = best_final_clf

        # Pickling classifier and CountVectorizers as one file with all meta data
        export_model_file(best_final_clf)

            
    # Print final classifier scores for all categories
    print('='*100,'\nFINAL RESULTS:')
    print_selected_models(selected_models_by_category, print_all = True)

    # Pickling classifiers and CountVectorizers as one file with all meta data
    #export_model_files(selected_models_by_category)

    print("Total process time: %g s" % (time.time()- start))


def main_2():
    start = time.time()
    ## Get input arguments
    (options, args) = get_args()

    ## Fetch input data to be classified
    #train_data_df = get_data_df(options, args)
    train_data_df = get_data_df_2(options, args)


    #sys.exit()
    print(train_data_df.shape)
    print(train_data_df.columns)
    
    # Result holder for all categories
    selected_models_by_category = {}

    # Get unique list of categories with ids
    categories = pandas.unique(train_data_df.loc[:,['category_full_path_mod1','category_id_mod1']].values)

    # Indicate classifiers to be tested
    clf_dict = {}
    #clf_dict['KNeighbors']          = KNeighborsClassifier()
    #clf_dict['Logistic Regression'] = LogisticRegression(penalty='l2')
    #clf_dict['Decision Tree']       = tree.DecisionTreeClassifier()
    #clf_dict['Random Forest']       = RandomForestClassifier()
    #clf_dict['MultinomialNB']       = MultinomialNB()
    #clf_dict['SVM']                 = svm.SVC(probability=True)
    #clf_dict['AdaBoost']            = AdaBoostClassifier()
    

    one_class_clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

    # Start iterating categories through classifiers
    for category in categories:    
        print('='*100,'\nRun models for following category:\n', category[0], '\n', '='*100)

        # Get (positives, negatives) for training set
        (positives, negatives) = get_positives_negatives(train_data_df, category[0])

        # Create training set
        print('   Vectorizing training data ...')
        ng_range = (1,1)
        count_vect = CountVectorizer(min_df=1, ngram_range=ng_range, binary = False)
        tfidf_vect = TfidfVectorizer(sublinear_tf=True, max_df=1.0, ngram_range=ng_range, stop_words='english')
        (X_train_counts, Y), count_vect_fitted = get_vectorized_data(positives, negatives, count_vect=count_vect, tfidf_vect=None)
        X_train_counts_pos = count_vect.fit_transform(pandas.concat([positives]))


        print('   Training classifier ...')
        one_class_clf.fit(X_train_counts_pos)

        print('   Predicting ...')
        X_train_counts = count_vect.transform(pandas.concat([positives,negatives]))
        predictions = one_class_clf.predict(X_train_counts)

        print('   Prediction accuracy score: ', accuracy_score(Y, predictions))
        print('\nConfusion matrix:')
        display(pandas.crosstab(pandas.Series(Y), predictions, rownames=['True'], colnames=['Predicted'], margins=True))


if __name__ == "__main__":
    main_1()  # runs regular multi-class classifiers
    #main_2()  # runs one-class svm




 
