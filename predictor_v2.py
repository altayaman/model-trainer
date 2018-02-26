import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
import psycopg2  ## for RedShift
import sys
import getpass
from optparse import OptionParser
from configparser import SafeConfigParser
from multiprocessing import Pool
from functools import partial
from IPython.display import display, HTML
#import configparser
import time
import pickle
# ML classifier libraries
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# Aux libraries for ML classifiers
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer

#global_db_engine = None


## Read arguments
def get_args():
    help_text = """CME Node Recall Booster"""
    parser = OptionParser(usage=help_text)


    # MsSQL credentials
    parser.add_option("-H",    "--host",           dest="host",            default = "", help="the url for the DB", metavar="HOST_ADDRESS")
    parser.add_option("-d",    "--db",             dest="database",        default = "",                       help="the logical db name")
    parser.add_option("-u",    "--username",       dest="username",        default = "",             help="the username for the DB", metavar="NAME")
    
    parser.add_option("-p",    "--password",       dest="password")
    parser.add_option("-m",    "--model",          dest="model",           help="file that contains classifier, countvectorizer and their meta data")
    parser.add_option("-i",    "--input_table",    dest="input_table")
    parser.add_option("-o",    "--output_table",   dest="output_table")
    parser.add_option("--pkl", "--pickle",         dest="pickle",          help="the pickle file if data is cached", metavar="PICKLE")
    parser.add_option("--cf",  "--config_file",    dest="config_file",     help="config_file that has all settings")

    (options, args) = parser.parse_args()
    # if options.output_table is not None and options.model is not None and (options.pickle is not None or options.input_table is not None):
    #     password = getpass.getpass("\nPlease enter the password for the DB: ")
    #     options.password = password

    #     print("\nEstablished connection with the database")
    #     global global_db_engine
    #     global_db_engine = get_db_engine(options)
    #     return (options, args)
    # else:
    #     print("Need to specify Output table and model file and one of the followings:")
    #     print('Input table')
    #     print('OR')
    #     print('Pickled input data')
    #     sys.exit()

    return (options, args)

def check_if_file_exists(file_path):
    try:
        with open(file_path) as infile:
            pass
    except IOError:
        print('  ERROR: '+file_path+' file not found\n')
        sys.exit()

def get_db_engine_2(config_set_, username_, password_, host_, database_, port_):
    # for RedShift and Post
    if(config_set_ == 'redshift'):
        url = ''.join(['postgresql://', username_, ":", password_, "@", host_, ':',port_, '/', database_])

    # for MsSQL
    elif(config_set_ == 'mssql'):
        url = ''.join(['mssql+pymssql://', username_, ":", password_, "@", host_, ':',port_, '/', database_])

    print()
    print('='*100)
    print('%-22s' % 'Connecting to url',':',url)
    engine = create_engine(url)
    return engine

def get_db_engine(options):
    # for RedShift
    #url = "".join(["postgresql://", options.username, ":", options.password, "@", options.host, ":5439/", options.database])

    # for MsSQL
    url = "".join(["mssql+pyodbc://", options.username, ":", options.password, "@", options.host, ":1143/", options.database])

    engine = create_engine(url)
    return engine

def check_n_get_config_options(options_, args):
    parser = SafeConfigParser()

    if(options_.config_file):
        parser.read(options_.config_file)
        prediction_option_ls = parser.options('prediction')

        options_ls_1 = ['test_data', 'model_file', 'prediction_dest', 'pickle']
        options_ls_2 = ['test_data', 'model_file', 'prediction_dest']
        options_ls_3 = ['pickle','model_file','prediction_dest']
        
        # retuns following values
        # parser, test_data_arg_section, test_data_arg, model_file_name, prediction_arg_section, prediction_arg, pickle_file_name
        if(set(prediction_option_ls).issuperset(set(options_ls_1))):
            return parser,  \
                   (parser.get('prediction', 'test_data')).split('+')[0],        \
                   (parser.get('prediction', 'test_data')).split('+')[1],        \
                   parser.get('prediction', 'model_file'),                       \
                   (parser.get('prediction', 'prediction_dest')).split('+')[0],  \
                   (parser.get('prediction', 'prediction_dest')).split('+')[1],  \
                   parser.get('prediction', 'pickle')
        elif(set(prediction_option_ls).issuperset(set(options_ls_2))):
            return parser,  \
                   (parser.get('prediction', 'test_data')).split('+')[0],  \
                   (parser.get('prediction', 'test_data')).split('+')[1],   \
                   parser.get('prediction', 'model_file'),                 \
                   (parser.get('prediction', 'prediction_dest')).split('+')[0],  \
                   (parser.get('prediction', 'prediction_dest')).split('+')[1],  \
                   None
        elif(set(prediction_option_ls).issuperset(set(options_ls_3))):
            return parser,   \
                   None,     \
                   None,     \
                   parser.get('prediction', 'model_file'),      \
                   (parser.get('prediction', 'prediction_dest')).split('+')[0],  \
                   (parser.get('prediction', 'prediction_dest')).split('+')[1],  \
                   parser.get('prediction', 'pickle')
        else:
            print('Some prediction argumnets are missing')
            sys.exit()
    else:
        print(' --cf option is not passed')
        sys.exit()

def get_DBapi_result_proxy(parser, test_data_arg_section, test_data_arg, pickle_file):

    if(test_data_arg_section == 'config_file'):
        test_file_name = parser.get(test_data_arg_section, test_data_arg)
        check_if_file_exists(test_file_name)
        df = pd.read_csv(test_file_name)
        print('%-22s' % ' Test data source' ,':',test_file_name)
        print('%-22s' % ' Size of dataframe',':',df.shape)

        if parser.get('prediction', 'pickle'):
            df.to_pickle(parser.get('prediction', 'pickle'))
            print('%-22s' % '\n Test data pickled as',':',parser.get('prediction', 'pickle'))

    #elif(test_data_arg_section == 'config_redshift'):
    elif(parser.get(test_data_arg_section, 'db_type') == 'redshift'):
        # Get database engine
        db_type  = parser.get(test_data_arg_section, 'db_type')
        username = parser.get(test_data_arg_section, 'username')
        password = parser.get(test_data_arg_section, 'password')
        host     = parser.get(test_data_arg_section, 'host')
        database = parser.get(test_data_arg_section, 'database')
        port     = parser.get(test_data_arg_section, 'port')
        engine   = get_db_engine_2(db_type, username, password, host, database, port)
        print(" Established connection with the database")

        # Fetch data from db
        start = time.time()
        print("\n Getting ResultProxy ...")
        test_data_table_name = parser.get(test_data_arg_section, test_data_arg)
        #df = pd.read_sql_query('SELECT * FROM ' + test_data_table_name + ' limit 3', engine)
        qry = 'SELECT * FROM ' + test_data_table_name #+ ' limit 3'
        conn = engine.connect()
        resultProxy = conn.execute(qry)

        ## Get elapsed time
        end = time.time()
        print(" Getting ResultProxy took %g s" % (end - start))
        print('%-22s' % ' Test data source' ,':',test_data_table_name,' from RedShift')
        #print(" Found " + str(len(df)) + " entries")

    #elif(test_data_arg_section == 'config_mssql'):
    elif(parser.get(test_data_arg_section, 'db_type') == 'mssql'):
        # Get database engine
        db_type  = parser.get(test_data_arg_section, 'db_type')
        username = parser.get(test_data_arg_section, 'username')
        password = parser.get(test_data_arg_section, 'password')
        host     = parser.get(test_data_arg_section, 'host')
        database = parser.get(test_data_arg_section, 'database')
        port     = parser.get(test_data_arg_section, 'port')
        engine   = get_db_engine_2(db_type, username, password, host, database, port)
        print(" Established connection with the database")

        # Fetch data from db
        start = time.time()
        print("\n Getting ResultProxy ...")
        test_data_table_name = parser.get(test_data_arg_section, test_data_arg)
        #df = pd.read_sql_query('SELECT top 3 * FROM ' + test_data_table_name, engine)
        qry = 'SELECT top 3 * FROM ' + test_data_table_name
        conn = engine.connect()
        df = conn.execute(qry)
        resultProxy = conn.execute(qry)

        ## Get elapsed time
        end = time.time()
        print(" Getting ResultProxy took %g s" % (end - start))
        print('%-22s' % ' Test data source' ,':',test_data_table_name,' from MsSQL')
        #print(" Found " + str(len(df)) + " entries")

    #elif parser.get('prediction', 'pickle'):
    else:
        #pkl_file = parser.get('prediction', 'pickle')
        #check_if_file_exists(pkl_file)
        check_if_file_exists(test_data_arg_section)
        #df = pd.read_pickle(pkl_file)
        df = pd.read_pickle(test_data_arg_section)
        #print('%-22s' % ' Test data source',':',pkl_file)
        print('%-22s' % ' Test data source',':',test_data_arg_section)
        print('%-22s' % ' Size of dataframe',':',df.shape)

    return resultProxy

def get_data_df_2(parser, test_data_arg_section, test_data_arg, pickle_file):

    if(test_data_arg_section == None):
        #pkl_file = parser.get('prediction', 'pickle')
        #check_if_file_exists(pkl_file)
        check_if_file_exists(pickle_file)
        #df = pd.read_pickle(pkl_file)
        df = pd.read_pickle(pickle_file)
        #print('%-22s' % ' Test data source',':',pkl_file)
        print('%-22s' % 'Test data source',':',pickle_file)
        print('%-22s' % 'Size of dataframe',':',df.shape)

    elif(test_data_arg_section == 'config_file'):
        test_file_name = parser.get(test_data_arg_section, test_data_arg)
        check_if_file_exists(test_file_name)
        df = pd.read_csv(test_file_name)
        print('%-22s' % 'Test data source' ,':',test_file_name)
        print('%-22s' % 'Size of dataframe',':',df.shape)

        if parser.get('prediction', 'pickle'):
            df.to_pickle(parser.get('prediction', 'pickle'))
            print('%-22s' % '\n Test data pickled as',':',parser.get('prediction', 'pickle'))

    #elif(test_data_arg_section == 'config_redshift'):
    elif(parser.get(test_data_arg_section, 'db_type') == 'redshift'):
        # Get database engine
        db_type  = parser.get(test_data_arg_section, 'db_type')
        username = parser.get(test_data_arg_section, 'username')
        password = parser.get(test_data_arg_section, 'password')
        host     = parser.get(test_data_arg_section, 'host')
        database = parser.get(test_data_arg_section, 'database')
        port     = parser.get(test_data_arg_section, 'port')
        engine   = get_db_engine_2(db_type, username, password, host, database, port)
        print("Established connection with the RedShift")

        # Fetch data from db
        start = time.time()
        print("\nReading input test data from RedShift ...")
        test_data_table_name = parser.get(test_data_arg_section, test_data_arg)
        df = pd.read_sql_query('SELECT * FROM ' + test_data_table_name, engine)
        #print('SELECT * FROM ' + test_data_table_name)

        ## Get elapsed time
        end = time.time()
        print("Reading data from RedShift took %g s" % (end - start))
        print("Found " + str(len(df)) + " entries")

        #if parser.get('prediction', 'pickle'):
        if pickle_file:
            #print(parser.get('prediction', 'pickle'))
            #df.to_pickle(parser.get('prediction', 'pickle'))
            df.to_pickle(pickle_file)
            #print('%-22s' % 'Test data pickled as',':',parser.get('prediction', 'pickle'))
            print('%-22s' % '\n Test data pickled as',':',pickle_file)

    elif(parser.get(test_data_arg_section, 'db_type') == 'mssql'):
        # Get database engine
        db_type  = parser.get(test_data_arg_section, 'db_type')
        username = parser.get(test_data_arg_section, 'username')
        password = parser.get(test_data_arg_section, 'password')
        host     = parser.get(test_data_arg_section, 'host')
        database = parser.get(test_data_arg_section, 'database')
        port     = parser.get(test_data_arg_section, 'port')
        engine   = get_db_engine_2(db_type, username, password, host, database, port)
        print("Established connection with the MsSQL")

        # Fetch data from db
        start = time.time()
        print("\nReading input test data from MsSQL ...")
        test_data_table_name = parser.get(test_data_arg_section, test_data_arg)
        df = pd.read_sql_query('SELECT * FROM ' + test_data_table_name, engine)
        #print('SELECT * FROM ' + test_data_table_name)

        ## Get elapsed time
        end = time.time()
        print("Reading data from MsSQL took %g s" % (end - start))
        print("Found " + str(len(df)) + " entries")

        #if parser.get('prediction', 'pickle'):
        if pickle_file:
            #print(parser.get('prediction', 'pickle'))
            #df.to_pickle(parser.get('prediction', 'pickle'))
            df.to_pickle(pickle_file)
            #print('%-22s' % 'Test data pickled as',':',parser.get('prediction', 'pickle'))
            print('%-22s' % '\nTest data pickled as',':',pickle_file)

    #elif parser.get('prediction', 'pickle'):
    # else:
    #     #pkl_file = parser.get('prediction', 'pickle')
    #     #check_if_file_exists(pkl_file)
    #     check_if_file_exists(test_data_arg_section)
    #     #df = pd.read_pickle(pkl_file)
    #     df = pd.read_pickle(test_data_arg_section)
    #     #print('%-22s' % ' Test data source',':',pkl_file)
    #     print('%-22s' % ' Test data source',':',test_data_arg_section)
    #     print('%-22s' % ' Size of dataframe',':',df.shape)

    #else:
    #    print(" Need to specify either input db table or pickle file or input file")
    #    sys.exit()

    return df

def get_data_df(options, args):
    if options.input_table is not None:
        # Fetch data from db
        start = time.time()
        print("\nReading input data from database ...")
        df = pd.read_sql_query('SELECT description_mod1 FROM ' + options.input_table, global_db_engine)

        ## Get elapsed time
        end = time.time()
        print("Reading input data from database took %g s" % (end - start))
        
        if options.pickle is not None:
            df.to_pickle(options.pickle)
    elif options.input_table is None and options.pickle is not None:
        df = pd.read_pickle(options.pickle)

    print("Found " + str(len(df)) + " entries")
    return df


#def get_model_file_2(options_):
def get_model_file_2(model_file_name):
    # parser = SafeConfigParser()

    # if(options_.config_file):
    #     parser.read(options_.config_file)
    # else:
    #     print(' ERROR: Config file not passed as --cf argument')
    #     sys.exit()

    # if(parser.get('prediction', 'model_file')):
    #     model_file_name = parser.get('prediction', 'model_file')
    #     check_if_file_exists(model_file_name)
    # else:
    #     print(' ERROR: model file not indicated')

    check_if_file_exists(model_file_name)

    with open(model_file_name, 'rb') as pickle_file:
        model_file = pickle.load(pickle_file)
    return model_file

def get_model_file(options_):
    with open(options_.model, 'rb') as pickle_file:
        model_file = pickle.load(pickle_file)
    return model_file

def prediction_to_str(clf_prediction, category_id):
    if(clf_prediction == 1):
        return str(category_id)
    else:
        return 'not ' + str(category_id)

def predict(description_str, clf, count_vect, category_id):
    # Vectorize description string
    row_vectorized = count_vect.transform([str(description_str)])

    # Get prediction
    clf_prediction = clf.predict(row_vectorized)

    # Prediction to string
    clf_prediction_str = prediction_to_str(clf_prediction, category_id)

    return clf_prediction_str

def predict_proba(description_str, clf, count_vect):
    # Vectorize description string
    row_vectorized = count_vect.transform([str(description_str)])

    # Get prediction probability
    clf_prediction_proba = clf.predict_proba(row_vectorized)[0][1]

    return clf_prediction_proba   

def get_predictions(input_data_df, model_file):
    ## Get category id, classifier and CountVectorizer from model file
    category_id = model_file['Category ID']
    clf         = model_file['Model'] 
    count_vect  = model_file['Count vectorizer']

    ## Get description column name from data frame
    col_name = input_data_df.columns.values[0]

    print('\nClassification is in progress ...')
    start = time.time()

    ## Add 2 columns with predictions and prediction probabilities
    input_data_df['Prediction']       = input_data_df[col_name].apply(lambda x: predict(x, clf, count_vect, category_id))
    input_data_df['Prediction_proba'] = input_data_df[col_name].apply(lambda x: predict_proba(x, clf, count_vect))

    ## Get elapsed time
    end = time.time()
    print("Classification took %g s" % (end - start))

    return input_data_df

def apply_predictions(clf, count_vect, category_id, col_name, input_data_df):
    ## Add 2 columns with predictions and prediction probabilities
    start = time.time()
    print('      Prediction thread ...')
    input_data_df['Prediction']       = input_data_df[col_name].apply(lambda x: predict(x, clf, count_vect, category_id))
    input_data_df['Prediction_proba'] = input_data_df[col_name].apply(lambda x: predict_proba(x, clf, count_vect))
    print("      Prediction thread took %g s" % (time.time() - start))

    return input_data_df

def get_predictions_parallel(input_data_df, model_file, num_partitions = 2, num_cores = 1):
    ## Get category id, classifier and CountVectorizer from model file
    category_id = model_file['Category ID']
    clf         = model_file['Model'] 
    count_vect  = model_file['Count vectorizer']

    ## Get description column name from data frame
    col_name = input_data_df.columns.values[0]

    print('\nClassification is in progress ...')
    start = time.time()

    ## Splitting test data into chunks
    #print('   Splitting test data into ' + str(num_partitions) + ' chunks ...')
    input_data_df_split = np.array_split(input_data_df, num_partitions)

    ## Get predictions for each split multithreaded in multiple pools
    print('   Parallelizing predictions ...')
    apply_predictions_partial = partial(apply_predictions, clf, count_vect, category_id, col_name)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(apply_predictions_partial, input_data_df_split))
    pool.close()
    pool.join()

    ## Get elapsed time
    end = time.time()
    print("Classification took %g s" % (end - start))

    return df

def get_ranges_for_df(input_df_size_, insertion_chunk_size_):
    ranges_ = []

    c = 0
    while(True):
        if(input_df_size_ >= insertion_chunk_size_):
            input_df_size_ = input_df_size_ - insertion_chunk_size_
            range_ = (c * insertion_chunk_size_ , (c+1) * insertion_chunk_size_ - 1)
            ranges_.extend([range_])
            c = c + 1
            if(input_df_size_ == 0):
                break
        else:
            if(input_df_size_-1 < c*insertion_chunk_size_):
                range_ = (c * insertion_chunk_size_, (c*insertion_chunk_size_) + input_df_size_ - 1)
            else:
                range_ = (c * insertion_chunk_size_, input_df_size_ - 1)
            ranges_.extend([range_])
            break

    return ranges_

def get_ranges_for_csv(input_df_size_, insertion_chunk_size_):
    ranges_ = [(0,0,insertion_chunk_size_)]
    input_df_size_ = input_df_size_ - insertion_chunk_size_
    c = 1
    while(True):
        if(input_df_size_ >= insertion_chunk_size_):
            input_df_size_ = input_df_size_ - insertion_chunk_size_
            #range_ = (c * insertion_chunk_size_ + 1, (c+1) * insertion_chunk_size_ - 1 + 2)
            range_ = (1, (c+1) * insertion_chunk_size_ + 1, insertion_chunk_size_)
            ranges_.extend([range_])
            c = c + 1
            if(input_df_size_ == 0):
                break
        else:
            if(input_df_size_-1 < c*insertion_chunk_size_):
                #range_ = (c * insertion_chunk_size_ + 1, (c*insertion_chunk_size_) + input_df_size_ - 1 + 2)
                range_ = (1, ((c)*insertion_chunk_size_) + 1, input_df_size_)
            else:
                #range_ = (c * insertion_chunk_size_ + 1, input_df_size_ - 1 + 2)
                range_ = (1, input_df_size_ + 1, 0)
            ranges_.extend([range_])
            break

    return ranges_

## -----------------------------------------------------------------------------------------------------------------------------

def insert_df_into_db_parallel(parser, prediction_arg_section, prediction_arg, insertion_chunk_size, predictions_df, df_partitions = 2, num_cores = 1):

    print('\nMain insertion of predicted results is in progress ...')
    start_time = time.time()

    insert_df_into_db_2_partial = partial(insert_df_into_db_2, parser, prediction_arg_section, prediction_arg, insertion_chunk_size)
    predictions_df_split = np.array_split(predictions_df, df_partitions)
    pool = Pool(num_cores)
    #df = pd.concat(pool.map(insert_df_into_db_2_partial, predictions_df_split))
    pool.map(insert_df_into_db_2_partial, predictions_df_split)
    pool.close()
    pool.join()

    ## Get elapsed time
    end_time = time.time()
    print('\nMain insertion of predicted results is complete ...')
    print("Main Insertion process took %g s" % (end_time - start_time))

def insert_df_into_db_2(parser, prediction_arg_section, prediction_arg, insertion_chunk_size, predictions_df):
    # Reset indexes
    predictions_df.reset_index(drop=True, inplace=True)

    # Create ranges for printing
    input_df_size = predictions_df.shape[0]

    # Get data-insert-portion ranges
    ranges = get_ranges_for_df(input_df_size, insertion_chunk_size)

    # Get db engine
    engine = None
    #if(prediction_arg_section == 'config_redshift'):
    if(parser.get(prediction_arg_section, 'db_type') == 'redshift'):
        # Get database engine
        db_type  = parser.get(prediction_arg_section, 'db_type')
        username = parser.get(prediction_arg_section, 'username')
        password = parser.get(prediction_arg_section, 'password')
        host     = parser.get(prediction_arg_section, 'host')
        database = parser.get(prediction_arg_section, 'database')
        port     = parser.get(prediction_arg_section, 'port')
        engine   = get_db_engine_2(db_type, username, password, host, database, port)
        print(" Established connection with RedShift")

    #elif(prediction_arg_section == 'config_mssql'):
    elif(parser.get(prediction_arg_section, 'db_type') == 'mssql'):
        # Get database engine
        db_type  = parser.get(prediction_arg_section, 'db_type')
        username = parser.get(prediction_arg_section, 'username')
        password = parser.get(prediction_arg_section, 'password')
        host     = parser.get(prediction_arg_section, 'host')
        database = parser.get(prediction_arg_section, 'database')
        port     = parser.get(prediction_arg_section, 'port')
        engine   = get_db_engine_2(db_type, username, password, host, database, port)
        print(" Established connection with MsSQL")

    conn = engine.connect()

    print('\nInsertion of predicted results is in progress ...')
    start_time = time.time()

    # Get prediction table name
    prediction_table_name = parser.get(prediction_arg_section, prediction_arg)

    ## Insert predictions into db by chunks
    for range_ in ranges:
        qry = "insert into " + prediction_table_name + " (description_mod1, predicted_category, probability) values "
        print(range_)
        for i in range(range_[0], range_[1]+1):
            comma = ","
            description = predictions_df.loc[i, 'description_mod1']
            description = description.replace("'","''")  # replace special characters
            description = description.replace(":","\:")  # replace special characters
            pred        = predictions_df.loc[i, 'Prediction']
            pred_proba  = predictions_df.loc[i, 'Prediction_proba']

            if(i == range_[0]):
                comma = ""

            line = comma + "('" + description + "','" + pred + "'," + str(pred_proba) + ')'
            qry = qry + line
        

        #print(qry)
        conn.execute(text(qry))

        ## commit
        #conn_RedShift.commit()
    
    ## close db connection
    conn.close()

    ## Get elapsed time
    end_time = time.time()
    print('\nInsertion of predicted results is complete ...')
    print("Insertion process took %g s" % (end_time - start_time))


def insert_df_into_db(insertion_chunk_size, input_data_df, options_):
    # Create ranges for printing
    input_df_size = input_data_df.shape[0]

    # Get data-insert-portion ranges
    ranges = get_ranges_for_df(input_df_size, insertion_chunk_size)

    print("\nConnects to DB to insert predicted results ...")
    ## connect to DB with credentials
    conn_RedShift = psycopg2.connect(host = 'host here',
                                     database = 'database here',
                                     port = 5439,
                                     user = 'username here', 
                                     password = 'password here')

    ## get cursor
    cur = conn_RedShift.cursor()


    print('\nInsertion of predicted results is in progress ...')
    start_time = time.time()

    for range_ in ranges:
        qry = "insert into " + options_.output_table + " (description_mod1, predicted_category, probability) values "
        print(range_)
        for i in range(range_[0], range_[1]+1):
            comma = ","
            description = input_data_df.loc[i, 'description_mod1']
            pred        = input_data_df.loc[i, 'Prediction']
            pred_proba  = input_data_df.loc[i, 'Prediction_proba']

            if(i == range_[0]):
                comma = ""

            line = comma + "('" + description.replace("'","''") + "','" + pred + "'," + str(pred_proba) + ')'
            qry = qry + line
        
        #print(qry)
        cur.execute(qry)

        ## commit
        conn_RedShift.commit()
    
    ## close db connection
    conn_RedShift.close()

    ## Get elapsed time
    end_time = time.time()
    print('\nInsertion of predicted results is complete ...')
    print("Insertion process took %g s" % (end_time - start_time))

def print_model_info(model_file_):
    model = model_file_['Model']
    print('Best params: ', str(model.best_params_))
    print('Best score: ', model.best_score_)
    print('Best estimator: ', model.best_estimator_)
    print('\n',model)


## =======================================================================================
def main_1():
    ## Get input arguments
    (options, args) = get_args()

    ## Get args from config file
    parser, test_data_arg_section, test_data_arg, model_file_name, prediction_arg_section, prediction_arg, pickle_file_name = check_n_get_config_options(options, args)
    print(test_data_arg_section, test_data_arg, model_file_name, prediction_arg_section, prediction_arg, pickle_file_name)

    ## Import model file which includes classifier and CountVectorizer
    #model_file = get_model_file(options)
    model_file = get_model_file_2(model_file_name)
    #print(model_file['Model'])

    ## Fetch input test data to be classified
    #input_data_df = get_data_df(options, args)
    test_data_df = get_data_df_2(parser, test_data_arg_section, test_data_arg, pickle_file_name)

    ## Get data with prediction results
    #predictions_df = get_predictions(test_data_df, model_file)
    predictions_df = get_predictions_parallel(test_data_df, model_file, num_partitions = 4, num_cores = None)

    ## Drop all predictions with probability < 0.7
    predictions_df = predictions_df.loc[predictions_df['Prediction_proba'] >= 0.7,:]
    print('Size of predictions after cutting : ', predictions_df.shape)

    ## Insert data with prediction results into db
    #insert_df_into_db(500, output_data_df, options)
    #insert_df_into_db_2(parser, prediction_arg_section, prediction_arg, 500, predictions_df)
    insert_df_into_db_parallel(parser, prediction_arg_section, prediction_arg, 500, predictions_df, df_partitions = 2, num_cores = None)


def main_2():
    ## Get input arguments
    (options, args) = get_args()

    ## Get args from config file
    parser, test_data_arg_section, test_data_arg, model_file_name, prediction_arg_section, prediction_arg, pickle_file_name = check_n_get_config_options(options, args)
    print(test_data_arg_section, test_data_arg, model_file_name, prediction_arg_section, prediction_arg, pickle_file_name)

    ## Import model file which includes classifier and CountVectorizer
    #model_file = get_model_file(options)
    model_file = get_model_file_2(model_file_name)
    #print_model_info(model_file)

    #sys.exit()


    ## 
    row_count = 39459897   # hard-coded rows count of csv file
    ranges = get_ranges_for_csv(row_count,10000)
    ranges = [ranges[0]]
    print(ranges)
    #sys.exit()


    # test_data_df = pd.read_csv('/Users/altay.amanbay/Desktop/infoprod_altay_testset_curr_tbl.csv',nrows=8, skiprows=range(0,0))
    # print(test_data_df)
    # sys.exit()

    for range_ in ranges:

        ## Fetch input test data to be classified
        #input_data_df = get_data_df(options, args)
        #test_data_df = get_data_df_2(parser, test_data_arg_section, test_data_arg, pickle_file_name)
        print(range_[0],range_[1])
        test_data_df = pd.read_csv('/Users/altay.amanbay/Desktop/infoprod_altay_testset_curr_tbl.csv',nrows=range_[2], skiprows=range(range_[0],range_[1]))
        test_data_df.rename(columns={'\ufeff"description_mod1"': 'description_mod1'}, inplace=True)
        #print(test_data_df.head())

        ## Get data with prediction results
        #predictions_df = get_predictions(test_data_df, model_file)
        predictions_df = get_predictions_parallel(test_data_df, model_file, num_partitions = 4, num_cores = None)
        #display(predictions_df.head())


        ## Insert data with prediction results into db
        #insert_df_into_db(500, output_data_df, options)
        #insert_df_into_db_2(parser, prediction_arg_section, prediction_arg, 500, predictions_df)
        insert_df_into_db_parallel(parser, prediction_arg_section, prediction_arg, 500, predictions_df, df_partitions = 2, num_cores = None)


def main_3():
    ## Get input arguments
    (options, args) = get_args()

    ## Get args from config file
    parser, test_data_arg_section, test_data_arg, model_file_name, prediction_arg_section, prediction_arg, pickle_file_name = check_n_get_config_options(options, args)
    print(test_data_arg_section, test_data_arg, model_file_name, prediction_arg_section, prediction_arg, pickle_file_name)


    # x = pd.HDFStore('store.h5', 'r')
    # #df = x['dframe'] # throws MemoryError

    # for df in x.select('dframe', chunksize = 1):
    #     print(df)
    # x.close()
    # sys.exit()

    ## HDF Store
    start = time.time()

    ## Create store.h5 file
    hdf_store = pd.HDFStore('store.h5', 'w', append = True)
    
    ## Get ResultProxy (cursor)
    resProxy = get_DBapi_result_proxy(parser, test_data_arg_section, test_data_arg, pickle_file_name)
    
    ## Get first row from db and column names as keys
    row = resProxy.fetchone()    
    resProxy_keys = resProxy.keys()
    
    ## Start fetching one row at a time and inserting into h5 file.
    ## Each time row is fetched, turned into df and is appended into hdf store.
    min_itemsize_ = {'description_mod1': 1000, 'category_id_mod1':5, 'category_full_path_mod1' : 150}
    c = 0
    while row:
        d = dict(row)
        df = pd.DataFrame(columns = resProxy_keys)
        for k in d.keys():
            df.loc[c,k] = str(d[k])
        c = c + 1
        hdf_store.append('dframe', df, data_columns = True, index = False, min_itemsize = min_itemsize_)
        row = resProxy.fetchone()


    ## Close h5 file and ResultProxy
    hdf_store.close()
    resProxy.close()

    end = time.time()
    print(" \nInserting data into HDF Store took %g s" % (end - start))

    sys.exit()  # HDF experimental



## main function
if __name__ == "__main__":
    start = time.time()

    main_1()

    end = time.time()
    print(" \nMain process took %g s" % (end - start))
    print('====================================================================================================================================')



