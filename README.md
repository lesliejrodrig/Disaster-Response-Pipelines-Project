# Disaster Response Pipeline Project

### Project Summary
In this project, you'll find a dataset provided by Figure Eight containing real messages that were sent during disaster events. You will be able to create a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This project will show off your software skills, including your ability to create basic data pipelines and write clean, organized code!

Below are a few screenshots of the web app.

### Project Components
There are three components you'll find in this project.<br/>
1. Data Folder<br/>
-A Python script, process_data.py, that writes a data cleaning pipeline that loads the messages and categories datasets, merges the two datasets, cleans the data, and stores it in a SQLite database (DisasterResponse.db)<br/>
-Disaster_Categories.csv<br/>
-Disaster_Messages.csv<br/>

2. Models Folder<br/>
-A Python script, train_classifier.py, writes a machine learning pipeline that loads data from the SQLite database splits the dataset into training and test sets, builds a text processing and machine learning pipeline, trains and tunes a model using GridSearchCV, outputs results on the test set, and exports the final model as a pickle file<br/>

3. App Folder<br/>
This is a Flask Web App that has data visualizations using Plotly and leverages the model created from our dataset. You can insert sample text to have it classified by our model and see the results.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
