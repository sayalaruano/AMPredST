# **AMPredST**

## **Table of contents:**

- [About the web application](#about-the-webapp)
- [Structure of the repository](#structure-of-the-repository)
- [Credits](#credits)
- [Further details](#details)
- [Contact](#contact)

## **About the project**

[Antimicrobial peptides](https://en.wikipedia.org/wiki/Antimicrobial_peptides) (AMPs) are small bioactive drugs, commonly with fewer than 50 amino acids, which have appeared as promising compounds to control infectious disease caused by multi-drug resistant bacteria or superbugs. These superbugs are not treatable with the available drugs because of the development of some mechanisms to avoid the action of these compounds, which is known as antimicrobial resistance (AMR). According to the World Health Organization, AMR is one of the [top ten global public health threats facing humanity in this century](https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance), so it is important to search for AMPs that combat these superbugs and prevent AMR.

**AMPredST** is a web application that allows users to predict the antimicrobial activity and general properties of AMPs using a machine learning-based classifier. The appication is based on a [previous project](https://github.com/sayalaruano/CapstoneProject-MLZoomCamp) that analyzed the best molecular descriptors and machine learning model to predict the antimicrobial activity of AMPs. The best model was `ExtraTreesClassifier` with max_depth of 50 and n_estimators of 200 as hyperparameters, and `Amino acid Composition` as the molecular descriptors.

<a href="https://ampredst.streamlit.app/" title="AMPredST"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"></a><br>

![ampredst-gif](./ampredst.gif)

## **Structure of the repository**

The main files and directories of this repository are:

|File|Description|
|:-:|---|
|[ExtraTreesClassifier_maxdepth50_nestimators200.zip](RandomForest_maxdepth10_nestimators200.zip)|Compressed file of the best classifier|
|[streamlit_app.py](streamlit_app.py)|Script for the streamlit web application|
|[requirements.txt](requirements.txt)|File with names of the packages required for the streamlit web application|
|[style.css](style.css)|css file to customize specific feature of the web application|

## **Credits**

- This project was inspired by the [notebook](https://github.com/dataprofessor/peptide-ml) and [video](https://www.youtube.com/watch?v=0NrFIGLwW0Q&feature=youtu.be) from [Dataprofessor](https://github.com/dataprofessor) about this topic.
- The [datasets](https://biocom-ampdiscover.cicese.mx/dataset), some ideas, and references to compare the performance of the best model were obtained from this [scientific article](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00251).

## **Further details**
More details about the exploratory data analysis, data preparation, and model selection are available in this [GitHub repository](https://github.com/sayalaruano/CapstoneProject-MLZoomCamp).


## **Contact**
[![](https://img.shields.io/twitter/follow/sayalaruano?style=social)](https://twitter.com/sayalaruano)

If you have comments or suggestions about this project, you can [open an issue](https://github.com/sayalaruano/ML_AMPs_prediction_streamlitapp/issues/new) in this repository, or email me at sebasar1245@gamil.com.
