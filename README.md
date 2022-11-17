# **Capstone Project of the Machine Learning ZoomCamp course**

## **Table of contents:**

- [About the project](#about-the-project)
- [Dataset](#dataset)
- [Data preparation and feature matrices](#data-preparation-and-feature-matrix)
- [Machine Learning Models](#machine-learning-models)
- [Results of the best ML model](#results-best-MLmodel)
- [Python virtual environment and installation of required libraries](#python-virtual-environment-and-installation-of-required-libraries)
- [How to run this app as a web service in a local server?](#how-to-run-this-app-as-a-web-service-in-a-local-server)
- [How to run this app as a web service in the cloud?](#how-to-run-this-app-as-a-web-service-in-the-cloud)
- [Structure of the repository](#structure-of-the-repository)
- [Contact](#contact)

## **About the project**

[Antimicrobial peptides](https://en.wikipedia.org/wiki/Antimicrobial_peptides) (AMPs) are small bioactive drugs, commonly with fewer than 50 amino acids, which have appeared as promising compounds to control infectious disease caused by multi-drug resistant bacteria or superbugs. These superbugs are not treatable with the available drugs because of the development of some mechanisms to avoid the action of these compounds, which is known as antimicrobial resistance (AMR). According to the World Health Organization, AMR is one of the [top ten global public health threats facing humanity in this century](https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance), so it is important to search for AMPs that combat these superbugs and prevent AMR.

However, the search for AMPs to combat superbugs by experimental methods is unfeasible because of the huge number of these compounds. So, it is required to explore other methods for handling this problem, and machine learning models could be great candidates for this task. Thus, in this project, I created some machine learning binary classifiers to predict the activity of antimicrobial peptides.

For this work, I took as a reference the [notebook](https://github.com/dataprofessor/peptide-ml) and [video](https://www.youtube.com/watch?v=0NrFIGLwW0Q&feature=youtu.be) from [Dataprofessor](https://github.com/dataprofessor) about this topic. Also, the datasets, some ideas, and references to compare the performance of the best model were obtained from this [article](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00251).

## **Dataset**

The [dataset](https://biocom-ampdiscover.cicese.mx/dataset) for this project consists of text files (in FASTA format) with sequences of active and non-active AMPs. The active AMPs were obtained by experimental assays, while the non-active peptides were derived from computational methods. The following table summarizes the dataset partitions and the number of instances in each one.

|Dataset partition|Size|AMPs|non-AMPs|
|:-:|---|---|---|
|Training|19548|9781|9767|
|Test|4656|2095|2561|
|External|13888|3117|10771|

AMPs can have more than one activity, including antibacterial, antifungal, antiparasitic, antiviral, among others. Training and Test partitions have active AMPs with a single activity, while the external partition has AMPs with more than one activity and it represents a real scenario of virtual screening with much more non-active AMPs than the active ones. You can find more details about these datasets in this [article](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00251).

Also, I chose these benchmark datasets to compare the performance metrics of our best models with the results of other methods reported in this [article](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00251).

## **Data preparation and feature matrices**

The feature matrices to train machine learning models were obtained by calculating some molecular features from the amino acid sequences of AMPs. These features were obtained with the [Pfeature](https://github.com/raghavagps/Pfeature) Python library.

For this work, I calculated ten features that require only input and output files as parameters. These features are summarized in the table below.

Feature class | Description | Python Function
---|---|---
AAC | Amino acid composition | aac_wp
ABC | Atom and bond composition | atc_wp, btc_wp
PCP | Physico-chemical properties | pcp_wp
RRI | Repetitive Residue Information | rri_wp
DDR | Distance distribution of residues |ddr_wp
SEP | Shannon entropy | sep_wp
SER | Shannon entropy of residue level | ser_wp
SPC | Shannon entropy of physicochemical property | spc_wp
CTC | Conjoint Triad Calculation | ctc_wp
CTD | Composition enhanced transition distribution | ctd_wp

To know the details of these features and the entire Python library, you can read the [Pfeature Manual](https://webs.iiitd.edu.in/raghava/pfeature/Pfeature_Manual.pdf).

Also, I used the [CD-HIT](https://github.com/weizhongli/cdhit) software to remove the redundant sequences of the AMPs.

You can find the code for this part in the **Feature matrices preparation** section of this [jupyter notebook](EDA_Binary_classifiers_AMPs_activity.ipynb).

## **Machine Learning Models**

First, I tested more than 30 ML binary classifiers using the [LazyPredict](https://github.com/shankarpandala/lazypredict) Python library. I chose the best models according to some performance metrics such as accuracy, ROC AUC, precision, recall, F1 score, and Matthews Correlation Coefficient (MCC). Then, I fine-tuned the hyperparameters of the best models using sklearn's class [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV). Finally, considering the results of hyperparameter tuning and performance metrics, I obtained the best ML model to predict AMPs activity.

You can find the code for this part in the **Machine learning models** section of this [jupyter notebook](EDA_Binary_classifiers_AMPs_activity.ipynb).

## **Results of the best ML model**

The best model was ExtraTreesClassifier with `max_depth` of 50 and `n_estimators` of 200 as hyperparameters (the others were set as default), and `Amino acid Composition` (aac_wp) as feature matrix.

The performance metrics of this model on test and external datasets are presented below.

Performance metric | Test dataset | External dataset
---|---|---
ROC AUC | 0.90 | 0.90
Accuracy | 0.90 | 0.92
Precision | 0.94 | 0.79
Recall | 0.85 | 0.86
F1score | 0.89 | 0.82
MCC | 0.81 | 0.77

According to the evaluation results of our best model on test and external datasets, it is performing quite well. This [article](https://pubs.acs.org/doi/full/10.1021/acs.jcim.1c00251) evaluated many the state of the art ML models for predicting AMPs using the same test and external datasets, and surprisingly the performance metrics of our model are very close to the results of the best ML models reported on this study.

You can find the code for this part in the **Machine learning models** section of this [jupyter notebook](EDA_Binary_classifiers_AMPs_activity.ipynb), or in this [Python script](train.py).

## **Python virtual environment and installation of required libraries**

I used [Pipenv](https://pypi.org/project/pipenv/) to create a Python virtual environment, which allows the management of python libraries and their dependencies. Each Pipenv virtual environment has a `Pipfile` with the names and versions of packages installed in the virtual environment, and a `Pipfile.lock`, a JSON file that contains versions of packages, and dependencies required for each package.

To create a Python virtual environment with libraries and dependencies required for this project, you should clone this GitHub repository, open a terminal, move to the folder containing this repository, and run the following commands:

```bash
# Install pipenv
$ pip install pipenv

# Create the Python virtual environment 
$ pipenv install

# Activate the Python virtual environment 
$ pipenv shell
```

You can find a detailed guide on how to use pipenv [here](https://realpython.com/pipenv-guide/).

However, if you use the Dockerfile of this project, you do not need to run these commands because Docker installs all Operative System and Python requirements, as is explained in the next section.

## **How to run this app as a web service in a local server?**

[Docker](https://www.docker.com/) allows to create **containers**, which are isolated environments with specific system requirements such as OS, libraries, programs, dependencies, among others. You can follow [instructions of the official documentation](https://docs.docker.com/engine/install/) to install this program, depending on your OS and other details.

The specifications of the docker container are stated in the `Dockerfile`, including the base image, instructions for installing libraries, files we need to copy from the host machine, and other instructions.

Once you have installed Docker, move to the folder containing this repository, open a terminal, and run the following commands:

```bash
# Build a docker image with specifications for this project stated in the Dockerfile
$ (sudo) docker build -t betalactamase-drug-discovery .

# Run the docker image 
$ (sudo) docker run -it --rm -p 9696:9696 betalactamase-drug-discovery
```

The screen-shot below shows how your terminal should look like after running the docker image:

<br />

<img src="Img/docker.png" width="800" height="150" alt="Docker image"/>

<br />

Then, you should open another terminal and run the `predict-test.py` python script, and you will obtain a prediction if the AMP defined with the `sequence` variable of the script will be active or non-active. The following screen-shot shows the expected result:

<br />

<img src="Img/predict.png" width="800" height="100" alt="Prediction"/>

<br />

If you want to make predictions on other AMPs, replace the `sequence` variable of the `predict-test.py` python script with the corresponding sequence.

## **How to run this app as a web service in the cloud?**

I used [Heroku](https://www.heroku.com/home) for hosting the web service in the cloud. For doing the deployment into the cloud, I followed [a tutorial](https://github.com/nindate/ml-zoomcamp-exercises/blob/main/how-to-use-heroku.md) provided by one of the members of the Machine Learning Zoomcamp, which has details and code required to deploy a web service to the cloud with Heroku.

If you want to try this option, you can replace the `url` variable of the `predict-test.py` python script with this link: https://amps-prediction.herokuapp.com/predict, as is shown in the following screen-shot:

<br />

<img src="Img/heroku.png" width="700" height="250" alt="Heroku"/>

<br />

Then, you need to run the `predict-test.py` python script in a terminal as in the last section, and you should obtain the same output without running the docker container locally.

## **Structure of the repository**

The main files and directories of this repository are:

|File/Folder|Description|
|:-:|---|
|[EDA_Binary_classifiers_AMPs_activity.ipynb](EDA_Binary_classifiers_beta_lactamase_drug_discovery.ipynb)|Jupyter notebook with EDA, feature matrices preparation, machine learning models, performance metrics of all models, and evaluation of the best model|
|[train.py](train.py)|Python script to train the best classifier|
|[predict.py](predict.py)|Python script to make predictions with the best classifier using a Flask's web service|
|[predict-test.py](predict-test.py)|Python script to send a request to the Flask's web service to make a prediction|
|[ExtraTreesClassifier_maxdepth50_nestimators200.zip](RandomForest_maxdepth10_nestimators200.zip)|Compressed file of the best classifier|
|[Dockerfile](Dockerfile)|Docker file with specifications of the docker container|
|[Pipfile](Pipfile)|File with names and versions of packages installed in the virtual environment|
|[Pipfile.lock](Pipfile.lock)|Json file that contains versions of packages, and dependencies required for each package|
|[Data/](Data/)|Original dataset, features matrices for training machine learning models obtained with Pfeature, and csv files with Amino acid composition feature matrix for training, test, and external datasets|
|[Output/](Output/)|Folder to save performance metrics and results of machine learning models|
|[Img/](Img/)|Folder to save images|

## **Contact**

If you have comments or suggestions about this project, you can [open an issue](https://github.com/sayalaruano/CapstoneProject-MLZoomCamp/issues/new) in this repository, or email me at sebasar1245@gamil.com.
