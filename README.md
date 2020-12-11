# Disaster Response Pipeline Project

This projet was intend to apply data engineering skills to analyze disaster data from [Figure Eight(Appen)](https://appen.com/figure-eight-is-now-appen/) and build a model for an API that classifies disaster messages.

In the Project Workspace contain real messages that were sent during disaster events. The challenge was to create a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also display visualizations of the data. This project will show off my software skills, including my ability to create basic data pipelines write clean and  organized code!

The project was started in 2020 by Gutelvam as a Udacity Nanodegree project.

Below are a few screenshots of the web app.

![graph](https://lh3.googleusercontent.com/pw/ACtC-3cXtSCG4s-MB2D-3W3l4QtSC40sq7esqZU-Ovq02QgeSebROtyjIcQsa6Mqhra00oyajCxKDcnGGLmStrWowcXAhpydPXrz8TGqAVoKJu98PzUZlSjiFR2PilmyIZuNhmApoV_BghIKK32AU-nwfDLt=w1218-h937-no?authuser=0)

![disaster-categories](https://lh3.googleusercontent.com/pw/ACtC-3dpqow-QKudII_1PJnUXDaT4JYvupuYApFzzFTFvPAbZEc55_F3nmV3EZksLLrzCSLoaCcbNp_6gckk8DQXanDVlMEwhKg2A1Hhr9SfvX25SdLH8plH2Frx8MH76gfPLynxR_3NW7zOKJrjXzPraPmV=w1032-h924-no?authuser=0)


<h4>Installation</h4>


***Dependencies***


Disaster response app requires:

        -Python (>= 3.6)

        -Flask(>= 1.1.2)
	
	    -numpy(>=1.18.4)

        -SciPy(>= 1.4.1)

	    -nltk(>=3.5)

        -SQLAlchemy(>= 1.3.20)

        -scikit-learn (>=0.23.1)

        -pandas(>=1.0.3)
	
	    -plotly(==4.14.1)

***User installation***


If you already have a working installation of  Anaconda, the easiest way to install all others packages like Flask, plotly and nltk by using pip:
 >pip install packagename

Otherwise you can use the comand below to install all requirements needed: 
 >pip install -r requirements.txt 

## Instructions of usage:
**1.** Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python ./data/process_data.py ./data/disaster_messages.csv ./data/disaster_categories.csv  ./data`

    - To run ML pipeline that trains classifier and saves
        `python ./models/train_classifier.py ./data/Disaster.db ./models`

**2.** Run the following command in the root directory to run your web app.
    `python ./app/run.py`

**3.** Go to http://127.0.0.1:3001/
	> if you have any trouble you can set your own port at: /app/run.py
	> app.run(host="127.0.0.1", port=3001, debug=True)
