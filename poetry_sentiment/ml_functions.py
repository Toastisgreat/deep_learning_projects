#initial imports
import pandas as pd
import re
from os import system, listdir
from os.path import isfile, join
from random import shuffle
import csv

#poetry reader


#data_framify
from nltk.stem import WordNetLemmatizer
from nltk import download
#vectorze_data
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from joblib import dump, load # used for saving and loading sklearn objects
from scipy.sparse import save_npz, load_npz # used for saving and loading sparse matrices

#train_and_show_scores
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

#best_learn
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

#RandomForestModel
from sklearn.ensemble import RandomForestClassifier
from random import randint




'''
TODO:
    fix gutenberg installation somehow bruh
'''
def get_files(fld: str) -> list:
        #input pos/neg folder, output list of files in folder
        return [join(fld, f) for f in listdir(fld) if isfile(join(fld, f))]

def append_files_data(data_list: list, file_path: str, collumn_list: list) -> None:
        '''
        Appends to 'data_list' tuples of form (file content, label)
        for each file in 'files' input list
        '''
        if file_path[-4:] == '.csv':     
            with open(file_path, 'r', encoding='latin-1') as f:
                text = csv.reader(f)
                for row in text:
                    data_list.append((row[collumn_list[0]], int(row[collumn_list[1]])))


        elif file_path[-4:] == '.tsv':
            with open(file_path, 'r', encoding='latin-1') as f:
                for row in f:
                    row = row.split('\t')
                    data_list.append((row[collumn_list[0]], int(row[collumn_list[1]])))


            print(f'extracted data from {file_path}')


def data_framify(folder: str, collumn_list: list, lemmatize: bool) -> list:  
    def lemmetizeword(word: str, lemmatizer: WordNetLemmatizer) -> str:
        try:
            return lemmatizer.lemmatize(word)
        except:
            download('wordnet')
            download('ome-1.4')
    def create_df(data: list, lemmatize = bool) -> pd.DataFrame:
        text, label = tuple(zip(*data))

        # replacing line breaks with spaces

        text = list(map(lambda txt: re.sub('(<br\s*/?>)+', ' ', txt), text))
        
        #lemmatize the data
        if lemmatize == True:
            lemmatized = WordNetLemmatizer()
            for line in text:
                for word in line:
                    word = lemmetizeword(word, lemmatized)

        return pd.DataFrame({'text': text, 'label': label})
    
    df_list = []
    files = get_files(folder)

    for file in files:
        data_list = []
        append_files_data(data_list, file, collumn_list)
        
        #randomize for better testing
        shuffle(data_list)
        df = create_df(data_list, lemmatize)
        df_list.append(df)

    return df_list
    
'''
def write_poetry_data(books: list, filepath: str) -> None:
    lineno = 0 
    with open(filepath, 'w') as file:

        for (id_nr, toskip, title) in books:

            #find text and do other stuff
            startline = lineno
            text = strip_headers(load_etext(id_nr)).strip()
            lines = text.split('\n')[toskip:]
            
            for line in lines:
                # any line that is all upper case is a title or author name
                # also don't want any lines with years (numbers)

                if len(line) > 0 and line.upper() != line and not re.match('.*[0-9]+.*', line) and len(line) < 50:
                    
                    #clean the line and write it to big poem file
                    clean_line = re.sub('[a-z/`/-]+','', line.strip().lower())
                    file.write(clean_line + '\n')
                    lineno += 1

                else:

                    #else, skip writing a line and just indicate that it's a new poem by skipping a line
                    file.write('\n')

            #indicate progress!
            print(f'Wrote lines {startline} to {lineno} from {title}')
'''
def make_dump(vectorframe: CountVectorizer, filepath: str) -> None:
    try:
        with open(filepath, 'w') as f:
            f.write('')
    except:
        pass
    dump(vectorframe, filepath)
def vectorize_data(dataset: pd.DataFrame, folder: str) -> list:
    #input dataset and mother folder name, output list of four dataframes

    #requires folders data_preprocessors and vectorized_data to already exist, will run error if fails
    #try/except exists for all of these to check if the data has already been made


    #Vectorizer process: create CountVectorizer object, fit dataset values, dump
    #Train process: transform vectorizer object with dataset values, save to file
    #TfIdf transformer process: create TdidfTransformer object, fit it with training variable, dump
    #Tfidf training process: use transformer to transform training data, save


    #unigram below
    try: 
        unigram_vectorizer = load(folder + 'data_preprocessors/unigram_vectorizer.joblib')
    except:
        unigram_vectorizer = CountVectorizer(ngram_range=(1,1))
        unigram_vectorizer.fit(dataset['text'].values)
        make_dump(unigram_vectorizer, folder + 'data_preprocessors/unigram_vectorizer.joblib')

    try:
        X_train_unigram = load_npz(folder + 'vectorized_data/X_train_unigram.npz')
    except:
        X_train_unigram = unigram_vectorizer.transform(dataset['text'].values)
        save_npz(folder + 'vectorized_data/X_train_unigram.npz', X_train_unigram)

    
    #unigram tf-idf below
    try:
        unigram_tf_idf_transformer = load(folder + 'data_preprocessors/unigram_tf_idf_transformer.joblib')
    except:
        unigram_tf_idf_transformer = TfidfTransformer()
        unigram_tf_idf_transformer.fit(X_train_unigram)
        make_dump(unigram_tf_idf_transformer, folder + 'data_preprocessors/unigram_tf_idf_transformer.joblib')
    
    try:
        X_train_unigram_tf_idf = load_npz(folder + 'vectorized_data/X_train_unigram_tf_idf.npz')
    except:
        X_train_unigram_tf_idf = unigram_tf_idf_transformer.transform(X_train_unigram)
        save_npz(folder + 'vectorized_data/X_train_unigram_tf_idf.npz', X_train_unigram_tf_idf)

    #bigram below
    try:
        bigram_vectorizer = load(folder + 'data_preprocessors/bigram_vectorizer.joblib')
    except:
        bigram_vectorizer = CountVectorizer(ngram_range=(1,2))
        bigram_vectorizer.fit(dataset['text'].values)
        make_dump(bigram_vectorizer, folder + 'data_preprocessors/bigram_vectorizer.joblib')

    try:
        X_train_bigram = load_npz(folder + 'vectorized_data/X_train_bigram.npz')
    except:
        X_train_bigram = bigram_vectorizer.transform(dataset['text'].values)
        save_npz(folder + 'vectorized_data/X_train_bigram.npz', X_train_bigram)
    
    #bigram tf-idf
    try:
        bigram_tf_idf_transformer = load(folder + 'data_preprocessors/bigram_tf_idf_transformer.joblib')
    except:
        bigram_tf_idf_transformer = TfidfTransformer()
        bigram_tf_idf_transformer.fit(X_train_bigram)
        make_dump(bigram_tf_idf_transformer, folder + 'data_preprocessors/bigram_tf_idf_transformer.joblib')
    
    try:
        X_train_bigram_tf_idf = load_npz(folder + 'vectorized_data/X_train_bigram_tf_idf.npz')
    except:
        X_train_bigram_tf_idf = bigram_tf_idf_transformer.transform(X_train_bigram)
        save_npz(folder + 'vectorized_data/X_train_bigram_tf_idf.npz', X_train_bigram_tf_idf)

    #trigram
    try:
        trigram_vectorizer = load(folder + 'data_preprocessors/trigram_vectorizer.joblib')
    except:
        trigram_vectorizer = CountVectorizer(ngram_range=(1,3))
        trigram_vectorizer.fit(dataset['text'].values)
        make_dump(trigram_vectorizer, folder + 'data_preprocessors/trigram_vectorizer.joblib')
    
    try:
        X_train_trigram = load_npz(folder + 'vectorized_data/X_train_trigram.npz')
    except:
        X_train_trigram = trigram_vectorizer.transform(dataset['text'].values)
        save_npz(folder + 'vectorized_data/X_train_trigram.npz', X_train_trigram)
    
    #trigram tf_idf
    try:
        trigram_tf_idf_transformer = load(folder + 'data_preprocessors/trigram_tf_idf_transofrmer.joblib')
    except:
        trigram_tf_idf_transformer = TfidfTransformer()
        trigram_tf_idf_transformer.fit(X_train_trigram)
        make_dump(trigram_tf_idf_transformer, folder + 'data_preprocessors/trigram_tf_idf_transformer.joblib')
    
    try:
        X_train_trigram_tf_idf = load_npz(folder + 'vectorized_data/X_train_trigram_tf_idf.npz')
    except:
        X_train_trigram_tf_idf = trigram_tf_idf_transformer.transform(X_train_trigram)
        save_npz(folder + 'vectorized_data/X_train_trigram_tf_idf.npz', X_train_trigram_tf_idf)

    return [X_train_unigram, X_train_unigram_tf_idf, X_train_bigram, X_train_bigram_tf_idf, X_train_trigram, X_train_trigram_tf_idf]

def train_and_show_scores(X: csr_matrix, y: np.array, title: str) -> float:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, train_size=0.75, stratify=y
        )

        #stochastic gradient descent classifier
        clf = SGDClassifier()
        clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        valid_score = clf.score(X_valid, y_valid)

        #SVM Model
        clf2 = SVC(kernel='linear')
        clf2.fit(X_train, y_train)
        train_score2 = clf2.score(X_train, y_train)
        valid_score2 = clf2.score(X_valid, y_valid)

        #Naive Bayes model
        clfGauss = MultinomialNB()
        clfGauss.fit(X_train, y_train)
        train_score3 = clfGauss.score(X_train, y_train)
        valid_score3 = clfGauss.score(X_valid, y_valid)

        print(f'{title}\nSGD Train score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\nSVM Train score : {round(train_score2, 2)} ; SVM Validation score : {round(valid_score2, 2)}\nNaives Bayes Train Score: {round(train_score3, 2)} ; Naive Bayes Validation score: {round(valid_score3, 2)}\n')
        
        return valid_score, valid_score2
    
def learn(XTU: pd.DataFrame, XTUT: pd.DataFrame, XTB: pd.DataFrame, XTBT: pd.DataFrame, XTT: pd.DataFrame, XTTT: pd.DataFrame, dataset: pd.DataFrame) -> None: 
    #input 4 df's from vectorize_data, output nothing
    
    y_train = dataset['label'].values

    train_and_show_scores(XTU, y_train, 'Unigram Counts')
    train_and_show_scores(XTUT, y_train, 'Unigram Tf-Idf')
    train_and_show_scores(XTB, y_train, 'Bigram Counts')
    train_and_show_scores(XTBT, y_train, 'Bigram Tf-Idf')
    train_and_show_scores(XTT, y_train, 'Trigram Counts')
    train_and_show_scores(XTTT, y_train, 'Trigram Tf-Idf')

def best_learn(X_train, dataset):
    clf = SGDClassifier()
    y_train = dataset['label']

    distributions = dict(
        loss=['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'],
        learning_rate=['optimal', 'invscaling', 'adaptive'],
        eta0=uniform(loc=1e-7, scale=1e-2)
    )

    random_search_cv = RandomizedSearchCV(
        estimator=clf,
        param_distributions=distributions,
        cv=5,
        n_iter=50
    )
    random_search_cv.fit(X_train, y_train)
    best_score = random_search_cv.best_score_
    best_params = random_search_cv.best_params_
    print(f'Best params(eta, learning rate, loss): {best_params}')
    print(f'Best score: {best_score}')

    clf = SGDClassifier()

    distributions = dict(
        penalty=['l1', 'l2', 'elasticnet'],
        alpha=uniform(loc=1e-6, scale=1e-4)
    )

    random_search_cv = RandomizedSearchCV(
        estimator=clf,
        param_distributions=distributions,
        cv=5,
        n_iter=50
    )
    random_search_cv.fit(X_train, y_train)
    best_score2 = random_search_cv.best_score_
    best_params2 = random_search_cv.best_params_
    print(f'Best params(alpha, penalty): {best_params2}')
    print(f'Best score: {best_score2}')

    return (best_params, best_params2)

#Random Forests Model
def random_forest_model(X: pd.DataFrame, y: pd.DataFrame, type: str) -> RandomForestClassifier:
    #test-train split
    X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, train_size=0.75, stratify=y
        )
    
    #create classifier and fit
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    print(f'{type} Scores:\nRandom Forest Train score: {round(rf.score(X_train, y_train), 2)} ; Validation score: {round(rf.score(X_valid, y_valid), 2)}')
    #find best params for classifier
    param_dist = {'n_estimators': range(50,500), 'max_depth': range(1,20)}
    rand_search = RandomizedSearchCV(rf, param_distributions = param_dist, n_iter=5, cv=5)
    rand_search.fit(X_train, y_train)
    best_rf = rand_search.best_estimator_

    print(f'Random Forest Best Train score: {round(best_rf.score(X_train, y_train), 2)} ; Best Validation score: {round(best_rf.score(X_valid, y_valid), 2)}')

    return best_rf


