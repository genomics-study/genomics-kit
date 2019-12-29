# Genomics Kit

This project was made as a part of an BSc thesis entitled: 

_"Ensemble of feature selection algorithms for multiclass classification of cardiovascular diseases"_

## Abstract

The purpose of the project was to apply machine learning algorithms for the feature extraction for 
a multiclass classification of genomic data, and
to develop method responsible for profitable combination of individual algorithms' results (ensemble learning).
Additionally a desktop application was implemented for end-user's convenience.

## Content

* `notebooks` directory containing open-ended research notebooks done though all stages
* `src` directory containing project's sources 
  * `gui` with desktop application sources
  * `model` containing core module
  * `preprocessing` module containing scripts for custom data simulation and cleaning HTN genomic data provided by client
  * `validation` module
  * `helpers` for miscellaneous utils

## Usage

Exemplary API usage can be found in `notebooks` directory e.g.: `model_example.ipynb`, `model_simulated_data.ipynb`

To run gui: `python3 gui.py`
