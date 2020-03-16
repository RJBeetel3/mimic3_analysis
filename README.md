# Machine Learning Nanodegree Capstone Project

Note: Currently converting what was a jupyter notebook based project to a script based project. The work is in progress. 

In this project a classifier was developed for predicting mortality for ICU patients given data from the first 24hrs of ICU admission. Patient data was collected from the the MIMIC-III (Medical Information Mart for Intensive Care III), a large, freely-available database comprising deidentified health-related data associated with patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012.
Data was pre-processed and features were selected. These features were used to train and test a number of candidate machine learning classifiers. Classifiers were optimized across their respective parameter spaces as well as across input feature set  and training/testing data set sizes. A classifier was selected from the group which provided the best performance with regard to predicting patient mortality. 

## Getting Started

Clone repo and install in a local directory using

`pip install -e . `
    
    
Data was queried from the mimic database in 3 groups, chart_events, lab_events and patient_demographics. Queries were saved as CSV files CHART_EVENTS_FIRST24.csv, LAB_EVENTS_FIRST24.csv and PTNT_DEMOG_FIRST24.csv. These groups of data were pre-processed separately in iPython notebooks with corresponding names, CHARTEVENTS_FIRST24.ipynb, LABEVENTS_FIRST24.ipynb and PATIENT_DEMOGRAPHICS_FIRST24.ipynb. Pre-processing  steps also included feature selection which used chi2 scores and corresponding p-values to select the features with the highest correlation with the outcomes. Selected features and corresponding feature selection scores were exported from each notebook and saved in the /features folder. 

A fourth iPython Notebook, ICU_MORTALITY_FIRST24.ipynb imports the selected features and scores, recombines and ranks the combined features and uses the top 20 to train, test and optimize candidate classifiers. 


### Prerequisites

Code was written in Python 2.7 installed using Anaconda


### Pre-Processing

** Note: The CHART_EVENTS_FIRST24.csv file is too large to store on github so is currently unavailable. 
CHARTEVENTS_FIRST24.ipynb will not run properly but the previously generated output files containing the 
chart_events features are in the repository and can be imported at the next stage. *** 

To generate the results from the raw .csv files, first run all code in the following three iPython notebooks: 

* CHARTEVENTS_FIRST24.ipynb
* LABEVENTS_FIRST24.ipynb 
* PATIENT_DEMOGRAPHICS_FIRST24.ipynb

These notebooks will generate the selected features and corresponding scores. The order in which they are run is not important


### Training, Testing and Optimizing Classifiers 
To complete feature selection and to train, test and optimize classifiers, run all code in: 

* ICU_MORTALITY_FIRST24.ipynb

**Note: While the code block that optimizes the rest of the candidate classifiers can be run in a reasonable amount of time, the block that optimizes the SVC classifier takes a VERY long time. Optimized classifiers, optimized parameters and classifier scores were exported using pickle.dump to Optimized_Classifiers.txt. Code for reading in the optimized classifier info can be found, commented out below the optimization blocks. If one were interested in saving time, one might skip the optimization code and simply upload the optimized classifier data.**

The output files from the pre-processing stages are included in the repository so one could begin directly with the ICU_MORTALITY_FIRST24.ipynb file

 

## Authors

* **Rob Beetel** - *Initial work* - [RJBeetel3](https://github.com/RJBeetel3)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Many thanks to the people who made and maintain the Mimic-III database without which none of this would have been possible. Very powerful things can be done with this type of data. 

Citations: MIMIC-III, a freely accessible critical care database. Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, and Mark RG. Scientific Data (2016). DOI: 10.1038/sdata.2016.35. Available at: http://www.nature.com/articles/sdata201635
Pollard, T. J. & Johnson, A. E. W. The MIMIC-III Clinical Database http://dx.doi.org/10.13026/C2XW26 (2016).

Also special thanks to the people at Yereva Research Labs who's project I looked to for guidance on feature selection/mining. 
https://github.com/YerevaNN
http://yerevann.com/
