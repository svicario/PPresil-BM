# PPresil-BM
##Overal description
Bayesian Harmonic  model for time series. 
For each year the program choose using Bayesian Factors among:
  
- the model of the year before
- a 3 harmonic model(yearly, semesterly and envery 3 month) with a priori parameter partially informed using the full time series
- a flat trend model.
   
 
Using the fitted model the program extract several yearly seasonal statistics and a single interyear standard variation statistics and their standard errors. 
In alternative is possible to obtain also the fitted value of the model with relative standard error  

##The annual statistics
The 3 annual statistics are estimated as default:   

- Annual Mean
- Annual Standard Deviation
- Annual Day of Maximum
   
Other statistics are optional:

- Day of the minimum and its value
- Value of the maximum
- Day of maximum change and its value expressed as difference between the two observed day around moment of maximum change

## The Input Format
The program expect as input file  a gdal readable geospatial raster dataset in which band name is formatted as time or as a string in which the last token is recognizable as a date, using as token separator the string "_".

PPresil options:
optional arguments:


    -h, --help            show this help message and exit
    --inputfile INPUTFILE
                          path to input file
    --suffix SUFFIX       suffix for output
    --step STEP           size of chunk to analyze
    -n N                  n-element of size stepxstep to analyze
    --noffset NOFFSET     offset to start counting n-element of size stepxstep
                          to analyze
    --change              allow change of seasonality
    --Expected            outout expected value and do not estimate annual
                          phenological statistics
    --Event               store the date and value of the maximum change within
                          each year
    --PP                  perform Posterior predictive test [experimental]
    --dense               assume that time series is dense and each year can be
                          fitted with its own seasonality model
    --freqsubsample FREQSUBSAMPLE
                          frequency of sampling from model in days used in order
                          to estimate phemolgical statistics
    --reps REPS           replicated of model sampling in order to estimate
                          standard variation of prediction
    --dask                replicated of model sampling in order to estimate
                          standard variation of prediction
    --netcdf              replicated of model sampling in order to estimate
                          standard variation of prediction
    --minmaxFeat          add to summary statistics min and max value and min
                          position
    --small               save on single file even if option n was used



## The Output Format 
The output format is a netcdf with as variables the statistics requested in input.
