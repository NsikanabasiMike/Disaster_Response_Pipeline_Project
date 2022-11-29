import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import accuracy_score, f1_score, \
recall_score, precision_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

from sqlalchemy import create_engine
import pickle

def load_data(database_filepath):
    """
    Gather data from sql database.
    
    Attributes: 
            None
    Returns: 
        string: array of plain text
        int: one hot encoded dataframe
    """
    
    engine = create_engine('sqlite:///'+database_filepath)
    
#     engine = create_engine(database_filepath)
    df = pd.read_sql('message_categories', con=engine)
    X = df.message.values
    Y = df.drop(['id', 'message', 'original','genre'], axis = 1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    
    """
    Split text into a list of tokens.
    
    Attributes: 
            array of plain text
    Returns: 
        list of tokenized words and numbers
    """
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens]
    return clean_tokens   


def build_model():
    
    """
    Build Machine Learning Pipeline.
    
    Attributes: 
            None
    Returns: 
        pipeline
    """
    
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, stop_words= 'english')),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))    
    ])
        
    return model

def evaluate_model(model, X_test, y_test, category_names):
    
    """
    Test pipeline.
    Prints f1_score, precision, and recall for each output category of the dataset.
    Parameters: 
            pipeline, test data
    Returns: 
        None
    """
   
    y_pred = model.predict(X_test)
    i= 0 
    for column in y_test:
        print(column+': '+ classification_report(y_test[column], y_pred[:,i]))
        i = i + 1
        
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro') 
    
    print('__________________________________________________________\n')
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

def improve_model(model):
    
    parameters = {
        'clf__estimator__n_estimators': [50,100, 200],
        'clf__estimator__min_samples_split': [2, 3]
    }
    
    cv = GridSearchCV(model, param_grid=parameters)
    return cv        

def display_results(cv, X_train, X_test, y_train, y_test):
#     cv.fit(X_train, y_train)
    y_pred = cv.predict(X_test)
    

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred) 

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("\nBest Parameters:", cv.best_params_)


def save_model(model,  model_filepath):
    
    """
    Loads pipeline to save to a pickle file.
    
    Parameters: 
           pipeine
    Returns: 
         None
    """
    
    with open(model_filepath, 'wb') as f:

        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, 'classifier.pkl')

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()