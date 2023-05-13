# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine, text
import re
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
        Load messages data from database.

    Inputs: 
        database_filepath: filepath of messages data in database
    Output:
        X (Series): target variable (messages column)
        y (Dataframe): feature variables (category columns)
        category_names: list of category types
    """
    # load messages dataframe
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages', engine)

    # split df into feature and target variables
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = list(Y.columns)

    return X, Y, category_names


def tokenize(text):
    """
        Clean text data by normalizing, tokenizing, lemmatizing and removing stop words.

    Inputs: 
        text: pre-processed string 
    Output:
        clean_tokens: list of cleaned and tokenized words
    """
    # replace urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # remove characters other than letters and numbers
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize text, remove whitespaces and convert to lowercase
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    
    # remove stopwords
    stop_words = list(set(stopwords.words('english')))
    clean_tokens = [token for token in clean_tokens if token not in stop_words]
    
    return clean_tokens


def build_model():
    """
        Build a machine learning pipeline using random forest classifier

    Inputs:
       None
    Output: 
        cv: GridSearch Model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier((RandomForestClassifier(n_estimators=10))))
    ])

    parameters = {
        'clf__estimator__min_samples_split': [2, 3],
        'clf__estimator__max_features': [0.25, 0.5]
    }

    cv = GridSearchCV(pipeline, parameters, verbose=2, cv=2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
        Report the f1 score, precision and recall for each category.
        
    Inputs:
       model: trained ML model
       X_test: Test data set of the message column
       Y_test: Test data set of the category values
       category_names: list of category types
    Output: 
        None
    """
    # Make model predictions
    Y_pred = model.predict(X_test)

    # Generate classification report for each category
    for i, col in enumerate(category_names):
        print(f'Category: {col}')
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i]))


def save_model(model, model_filepath):
    """
        Saves the model to a pickle file.
        
    Inputs:
       model: trained ML model
       model_filepath: filepath of where model will be saved
    Output: 
        None
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()