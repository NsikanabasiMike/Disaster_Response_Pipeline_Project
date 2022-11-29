## Table of Contents
- [Installation](#install)
- [Project Motivation](#motivate)
- [File Descriptions](#describe)
- [Instructions on How to Run the Files](#Instructions)
- [Licensing, Authors, and Acknowledgements](#acknowledge)

<a id='install'></a>
### Installation
1. Python 3.6 and latest from [here](https://www.python.org/downloads/)
2. Anaconda distribution of Python from [here](https://www.anaconda.com/blog/anaconda-distribution-2022-10#).
3. Natural Language Tool Kit (NLTK) and its functionalities like _'punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'_. These packages can be pip installed.

<a id='motivate'></a>
### Project Motivation
This project is about applying a combination of Software Engineering and Machine Learning(ML) skills to gather, clean, analyze, and build a ML classifier pipeline that classifies disaster messages. The disaster data is obtained from Appen (formally Figure 8). Details about the company [here](https://en.wikipedia.org/wiki/Figure_Eight_Inc.).<br>  

Datasets for this project contain real messages that were sent during disaster events. This project will build a machine learning pipeline to categorize these events so that messages can be sent to an appropriate disaster relief agency.
The project includes a web app where an emergency worker can input a new message and get classification results in several categories.<br>

The first part is the Extract, Transform, and Load(ETL) process. Here, the dataset is read into a dataframe, then cleaned with pandas, and stored in a SQLite database.

The second part is the machine learning portion. Here, the stored data is split into a training set and a test set. Then, a machine learning pipeline is built that uses Natural Language Tool Kit (NLTK), as well as scikit-learn's Pipeline to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). The model is exported to a pickle file.<br>

The third part is a basic web app that shows some graphs about the cleaned data. The graphs are plotted with plotly. It also collects data from a user and predicts the categories of such message.<br>



<a id='describe'></a>
### File Descriptions
There are three folders: app, data, and models.<br>
## Table of Contents
- [Installation](#install)
- [Project Motivation](#motivate)
- [File Descriptions](#describe)
- [Instructions on How to Run the Files](#Instructions)
- [Licensing, Authors, and Acknowledgements](#acknowledge)

<a id='install'></a>
### Installation
1. Python 3.6 and latest from [here](https://www.python.org/downloads/)
2. Anaconda distribution of Python from [here](https://www.anaconda.com/blog/anaconda-distribution-2022-10#).
3. Natural Language Tool Kit (NLTK) and its functionalities like _'punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'_. These packages can be pip installed.

<a id='motivate'></a>
### Project Motivation
This project is about applying a combination of Software Engineering and Machine Learning(ML) skills to gather, clean, analyze, and build a ML classifier pipeline that classifies disaster messages. The disaster data is obtained from Appen (formally Figure 8). Details about the company [here](https://en.wikipedia.org/wiki/Figure_Eight_Inc.).<br>  

Datasets for this project contain real messages that were sent during disaster events. This project will build a machine learning pipeline to categorize these events so that messages can be sent to an appropriate disaster relief agency.
The project includes a web app where an emergency worker can input a new message and get classification results in several categories.<br>

The first part is the Extract, Transform, and Load(ETL) process. Here, the dataset is read into a dataframe, then cleaned with pandas, and stored in a SQLite database.

The second part is the machine learning portion. Here, the stored data is split into a training set and a test set. Then, a machine learning pipeline is built that uses Natural Language Tool Kit (NLTK), as well as scikit-learn's Pipeline to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). The model is exported to a pickle file.<br>

The third part is a basic web app that shows some graphs about the cleaned data. The graphs are plotted with plotly. It also collects data from a user and predicts the categories of such message.<br>

<a id='describe'></a>
### File Descriptions
There are three folders: app, data, and models.<br>

__app__ is a python package which contains all the necessary files for the web app and visualization.

__data__ contains four files: __DisasterResponse.db__, __Disaster_messages.csv__, __Disaster_category.csv__, and __process_data.py__.<br>
__DisasterResponse.db__ is a sql database that contains the cleaned data.<br>
__Disaster_messages.csv__ is a csv file that contains disaster messages.<br>
__Disaster_category.csv__ is a csv file that contains disaster categories.<br>
__process_data.py__ is a python module that contains the code to gather, merge and process the data obtained from __Disaster_messages.csv__, and __Disaster_category.csv__.<br>

__models__ contains three files: __DisasterResponse.db__, __classifier.pkl__, and __train_classifier.py__<br>
__train_classifier.py__ is a python module that contains the code for ML pipeline<br>
__classifier.pkl__ is a pickle file that stores the trained ML pipeline.<br>


<a id='Instructions'></a>
### Instructions on How to Run the Files:

1. Run the following commands in the project's root directory to set up your database and model. On your command line, run:

    - To run ETL pipeline that cleans data and stores in database. 
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

<a id='acknowledge'></a>
### Licensing, Authors, Acknowledgements
The data used is from Appen in conjuction with Udacity.
