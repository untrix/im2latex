_________________
The Paper (Link TBD)   

_________________
### Test Results
Two model variations are presented:
#### I2L-NOPOOL
  1. [Random Sample of Predictions](./I2L-NOPOOL/rand_sample_100.html )
  2. [Correct Predictions](./I2L-NOPOOL/matched_strs_100.html)
  3. [Incorrect Predictions (Mistakes)](./I2L-NOPOOL/unmatched_rand_sample.html)
  4. [Attention Scan Visualization](./I2L-NOPOOL/alpha)
  5. Model Training Charts (Link TBD)
  
#### I2L-STRIPS
  1. [Random Sample of Predictions](./I2L-STRIPS/rand_sample_100.html)
  2. [Correct Predictions](./I2L-STRIPS/matched_strs_100.html)
  3. [Incorrect Predictions (Mistakes)](./I2L-STRIPS/unmatched_rand_sample.html)
  4. [Attention Scan Visualization](./I2L-STRIPS/alpha)
  5. Model Training Charts (Link TBD)

_________________
### Attention Scan Visualization (Both Models): 
[Examples](./alpha_index.html)

_________________
### Datasets
#### Formula Lists
Download these files if you want to generate your own dataset starting from normalized formulas (i.e. step1 of the preprocessing pipeline). You would copy this into file the folder named 'step0' (see the [preprocessing notebooks](https://github.com/untrix/im2latex/tree/master/src/preprocessing) for details). We recommend downloading the I2L-140K dataset since being larger of the two (and a superset of Im2latex-90K), it yields better generalization.
1. [I2L-140K Normalized Formula List (7MB Download)](https://storage.googleapis.com/i2l/data/dataset5/formulas.norm.filtered.txt.gz)
2. [Im2latex-90K Normalized Formula List (4MB Download)](https://storage.googleapis.com/i2l/data/dataset3/dataset3_step0.gz)

#### Full Dataset (Huge!)
All data starting with step0 through step5. Includes all images as well as all the DataFrames and numpy arrays produced at the end of the [data processing pipeline](https://github.com/untrix/im2latex/tree/master/src/preprocessing).

    WARNING: This is a HUGE download!

We recommend downloading the I2L-140K dataset since being larger of the two (and a superset of Im2latex-90K), it yields better generalization.
1. [I2L-140K (822MB Download)](https://storage.googleapis.com/i2l/data/dataset5.tgz)
2. [Im2latex-90K (542MB Download)](https://storage.googleapis.com/i2l/data/dataset3.tgz)

