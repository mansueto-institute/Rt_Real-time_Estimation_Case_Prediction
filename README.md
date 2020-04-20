# Bayesian Rt Estimation and Epidemiological New Case Prediction

This is a collection of Python Scripts to:

  - Take in epidemiological case timeseries (see below for sources)
  and
	- estimate the statistical distribution Effective Reproductive Number Rt in real time (mean and confidence interval)
	- predict the probability distribution - expected cases and confidence intervals - of future new cases.
  
The method was originally developed by [Bettencourt & Ribeiro (2008)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0002185) for emerging infectious diseases and recently beautifully implemented by [Kevin Systrom](https://twitter.com/kevin) and [Mike Krieger](https://twitter.com/mikeyk) for COVID-19 data as [Rt.live](Rt.live)

These scripts contain some improvements relative to the original methods as they use: 
- a fast parametric Bayesian update of standard (Susceptible-Infectious-Recovered/Dead) epidemiological models 
        with i) new cases described as before by a [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution), given previous cases,
        ii) the branching parameter bt described by a [Gamma distributed](https://en.wikipedia.org/wiki/Gamma_distribution) [conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior),        
        iii) the [posterior predictor](https://en.wikipedia.org/wiki/Posterior_predictive_distribution) of future cases described by a [Negative Binomial distribution](https://en.wikipedia.org/wiki/Negative_binomial_distribution);

- anomaly detection of new cases below or above the model's running expectation at some level of confidence (99% in the code)
- annealing (average preserving variance increase) of the Rt distribution to include the anomalous cases.
- windowing and some case smoothing (similar to the [Rt.live](Rt.live)
 implementation) to obtain local running estimates and avoid reporting chunking.

These developments make the method more robust, faster, more adaptive (avoid Bayesian overshrinking) and capable of handling different geographies, 
including nations, states, Metropolitan Areas and Counties, and other local areas with just a few daily cases. 

## Data 

Timeseries data for nations and US states can be downloaded from the [University of Washington Humanistic GIS Lab](https://hgis.uw.edu)  [COVID-19 page](https://hgis.uw.edu/virus/) in [csv](https://hgis.uw.edu/virus/assets/virus.csv)

Data for US Counties is compiled daily by the [New York Times](https://www.nytimes.com/interactive/2020/us/coronavirus-us-cases.html) and can be accessed [here](https://github.com/nytimes/covid-19-data)

### National and State Estimations

The code requires python 3 and the foillowing packages: csv, matplotlib, scipy, numpy, statsmodels.api

The nation or state of your choice is hard wired in Line 21 of the code 

```
nation='spain' # change to any other nation or US State, e.g. nation='arizona'
```

To run at a terminal:

```
python data_tracker_bayes.py
```
This will produce 3 images:

*The timeseries of new cases, showing also smoothing 

![Timeseries](Daily_New_Cases_w_Smoothing_Chicago-Naperville-Elgin IL-IN-WI.png)


*The running time estimation of Rt


*The running prediction of new cases (anomalies are shown as cases the  model did not initially predict well):

### Cities and Metropolitan Areas

[Metropolitan Statistical Areas](https://www.census.gov/programs-surveys/metro-micro/about.html) (MSAs) are the US Census definition of functional cities (integrated socioeconomic networks)

The code for Metropolitan Areas is similar. Choose your city in line 28 and produce the same three outputs.
```
nameMSA="Boston" # change to any other Metropolitan Area by writing its main city, e.g. nameMSA='Pheonix'
```

To run at a terminal:

```
python MSA_Epi_Bayesian_Estimation_Projection.py
```

Producing
<img src="./images/Daily_New_Cases_w_Smoothing_Chicago-Naperville-Elgin IL-IN-WI.png"> 

### US Counties

For US Counties, I organized them by State. Choosing a State reproduces the analysis for each county with more than a chose (here 5) number of cases:

Choose your State in line 28 and produce the same three outputs.
```
nameState="Illinois" # write the full name of your chosen state
```
To run at a terminal:

```
python County_Epi_Bayesian_Estimation_Projection.py
```

Beware: This produces many plots (3 per county in the state). Here are some examples from Illinois:


## Issues:

The code is very basic and can use your talent to make it better and more useful. Thank you.

## Authors

* [Luis M. A. Bettencourt](https://twitter.com/BettencourtLuis)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

