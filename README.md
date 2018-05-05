# Domains
## Preprocessing
### Data type detector
* Numeric
* Numeric-Categorical
* String-Categorical
* Time

### Binning
* Equal pupulation binning
* Equal value binning
* Monotonic binning

## Unsupervised learning
### Clustering
#### K-based clustering
#### Density-based clustering
#### Hierarchical clustering
#### Advanced clustering
* Spectral clustering
* Subspace clustering
* Multi-sourced clustering
* Multi-aspect clustering
* Multi-task clustering

#### Deep learning-based clustering
* AE + K-means
* AE + Spectral clustering
* AE + Subspace clustering

### Adversarial representation learning
* BiGAN
* infoGAN
* AAE


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