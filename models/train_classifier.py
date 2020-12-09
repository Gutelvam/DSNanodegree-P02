import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    df = pd.read_sql_table('clean_data', f'sqlite:///{database_filepath}') 
    X = df.message
    y = df.drop(columns= ['id', 'message', 'original', 'genre'])
    columns = list(y.columns.values)
    return X, y , columns


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
          ])
    return pipeline

def improve_model(model , Xtrain , ytrain):
    parameters = {
        'vect__min_df': [1, 5],
        'tfidf__use_idf':[True, False],
        'clf__estimator__n_estimators':[10, 25], 
        'clf__estimator__min_samples_split':[2, 5, 10]
        } 

    cv = GridSearchCV(model, param_grid=parameters, verbose = 10)
    cv.fit(Xtrain ,ytrain)

    return  cv


def evaluate_model(model, X_test, Y_test, category_names):

    pred = model.predict(X_test)
    ypred = pd.DataFrame(pred, columns = category_names).reset_index(drop= True)
    
    for row, col in enumerate(ypred):
        print(f"==============================> {col} <======================================")
        print('')
        print(classification_report(np.hstack(Y_test[col].values), np.hstack(ypred[col].values)))



def save_model(model, model_filepath):
    pickle.dump(model, open(f'{model_filepath}/classifier.pkl', 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model_cv = improve_model(model , X_train , Y_train)
        
        print('Evaluating model...')
        evaluate_model(model_cv, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model_cv, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()