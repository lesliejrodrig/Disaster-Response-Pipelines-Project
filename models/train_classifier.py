import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    """
    Load Data Function
    
    Input:
        database_filepath - Path to SQLite destination database (ex: disaster_response.db)
    Output:
        X - dataframe containing features
        Y - dataframe containing labels
        Y.columns - List of columns
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('MessagesCategories', engine)
    X = df['message']
    Y = df.drop(['id','message', 'original', 'genre'], axis=1)
    return X, Y, Y.columns


def tokenize(text):
    """
    Tokenize Text Function 
    
    Input:
        text - Text message to be tokenized
    Output:
        clean_tokens - List of tokens extracted from the text message
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build Model function
    Input:
        None
    Output:
        A GridSearchCV Pipeline
    """
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {'vect__max_features': (None, 5000), 
              'clf__estimator__n_estimators': [10, 20]}

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    This function applies a ML pipeline to a test set and prints out the model performance
    
    Input:
        pipeline - My GridSearchCV Pipeline
        X_test - Test features
        Y_test - Test labels
        category_names - label names
    """
    y_predict_test = model.predict(X_test)
    y_predict_df = pd.DataFrame(y_predict_test, columns = Y_test.columns)
    for column in Y_test.columns:
        print("Column Name: " + column)
        print(classification_report(Y_test[column],y_predict_df[column]))

    accuracy = (y_predict_test == Y_test).mean()
    print("Accuracy:", accuracy)
    pass


def save_model(model, model_filepath):
    """
    Save Model function
    
    Saves trained model as Pickle file
    
    Input:
        model - My GridSearchCV model
        model_filepath - path to save .pkl file
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


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