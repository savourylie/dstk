# Domains
## Inspection
### Bi-variant inspection
* chi2 (2 star)
* ANOVA (2 star)
* T-test (2 star)
* IV (3 star)
* KS (3 star)

### Check collinearity
* Collinearity
	* TBD
* Multicolinearity
	* Variance Inflation Factor (3 star)

### OOT inspection
* PSI (3 star)
* Dataframe comparison (unit tests-covered)

### Data type detector (3 star)
* Numeric
* Numeric-Categorical
* String-Categorical
* Time

## Preprocessing
### Imputing (unit tests-covered)
* Continous
	* mean
	* truncated mean
	* median
	* bin-nan
* Categorical
	* most frequent class
	* stringify

## Metric
### Response related metrics
### Clustering metrics

* Purity (need unit tests)
* Accuracy (need unit tests)

## Transformation
### Binning
* Equal pupulation binning (3 star)
* Equal value binning (3 star)
* Monotonic binning
* ChiMerge

### Encoding
* Dummy
* WOE
* Tree leaves encoding
 
## Clustering
### K-based clustering
### Density-based clustering
### Hierarchical clustering
### Advanced clustering
* Spectral clustering
* Subspace clustering
* Multi-sourced clustering
* Multi-aspect clustering
* Multi-task clustering

## Deep learning-based clustering
* AE + K-means
* AE + Spectral clustering
* AE + Subspace clustering

## Feature learning
### Adversarial representation learning
* BiGAN
* infoGAN
* AAE

<<<<<<< HEAD

# statistical description of raw data
> Exploring/Summarize the data distribution

## data type
> different processing methods for differnet types of data

### numeric
+ continuous: Data that can take on any value in an interval
+ discrete: Data that can only take on integer values

### categorical
> Data that can only take on a specific set of values

+ Binary: special case of categorical data, can only take two values

### ordinal
> Categorical data that has an explicit ordering


# Numeric data statistical description

## Estimates of Location
+ mean
+ truncated mean
+ weighted mean
+ median
+ outliers

## Estimates of Variability
+ variance: N-1
+ standard deviation
+ range: min/max values
+ percentiles
+ Interquartile Range(IQR): 75th percentile - 25th percentile

## data distribution exploration
+ Boxplot
+ Frequency table
+ histogram
+ density plot: kernal density estimate

# Categorical data statistical description
+ Mode: the most commonly category/value
+ Expected value: similar as weighted mean
+ Bar charts:The frequency or proportion for each category plotted as bars
+ Pie charts:The frequency or proportion for each category plotted as wedges in a pie
=======
### MISC
* Entity embeddings
>>>>>>> 33dbd805f63b3a24e7dc54b14f3599be9c91c4e0
