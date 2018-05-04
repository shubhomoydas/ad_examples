Python libraries required:
--------------------------
    numpy (1.13.3)
    scipy (0.19.1)
    scikit-learn (0.19.1)
    cvxopt
    pandas (0.21.0)
    ranking
    statsmodels
    matplotlib (2.1.0)
    tensorflow (1.6.0)


Anomaly Detection Examples
--------------------------
This is a collection of anomaly detection examples for detection methods popular in academic literature and in practice. I will include more examples as and when I find time.

Some techniques covered are listed below. These are a mere drop in the ocean of all anomaly detectors and are only meant to highlight some broad categories. Apologies if your favorite one is currently not included -- hopefully in time...
  - i.i.d setting:
    - [Standard unsupervised anomaly detectors](python/ad/ad_outlier.py) (Isolation Forest, LODA, One-class SVM, LOF)
    - [Clustering and density-based](python/ad/gmm_outlier.py)
    - [Density estimation based](python/ad/kde_outlier.py)
    - [PCA Reconstruction-based](python/ad/pca_reconstruct.py)
    - [Autoencoder Reconstruction-based](python/dnn/autoencoder.py)
    - [Classifier and pseudo-anomaly based](python/ad/pseudo_anom_outlier.py)
    - [Ensemble/Projection-based](python/loda/loda.py)
    - [A demonstration of outlier influence](python/ad/outlier_effect.py)
    - [Spectral-based](python/ad/spectral_outlier.py)
  - timeseries
    - Forecasting-based
      - [ARIMA](python/timeseries/timeseries_arima.py)
      - [Regression](python/timeseries/timeseries_regression.py) (SVM, Random Forest, Neural Network)
      - [Recurrent Neural Networks](python/timeseries/timeseries_rnn.py)
    - i.i.d
      - [Windows/Shingle based](python/timeseries/timeseries_shingles.py) (Isolation Forest, One-class SVM, LOF, Autoencoder)
  - human-in-the-loop (active learning)
    - [Active Anomaly Discovery](python/aad) -- see [section on AAD below](#active-anomaly-discovery-aad) for instructions on how to run.

There are multiple datasets (synthetic/real) supported. Change the code to work with whichever dataset or algorithm is desired. Most of the demos will output pdf plots under the 'python/temp' folder when executed.

**AUC** is the most common metric used to report anomaly detection performance. See [here](python/dnn/autoencoder.py) for a complete example with standard datasets.

The codebase also includes some [activity modeling stuff](#activity-modeling).

To execute the code:

1. **Run code from 'python' folder**. The outputs will be generated under 'temp' folder. The 'pythonw' command is used on OSX, but 'python' should be used on Linux.

2. *The run commands are at the top of the python source code files.*

3. Check the log file in **'python/temp'** folder. Usually it will be named <demo_code>.log. Timeseries demos will output logs under the 'python/temp/timeseries' folder.


Active Anomaly Discovery (AAD)
------------------------------
This codebase replaces the older 'pyaad' project (https://github.com/shubhomoydas/pyaad). It implements an algorithm (AAD) to actively explore anomalies. **Assuming that the ensemble scores have already been computed**, the file (https://github.com/shubhomoydas/ad_examples/blob/master/python/percept/percept.py) implements AAD in a much more simplified manner.

To run (https://github.com/shubhomoydas/ad_examples/blob/master/python/percept/percept.py):

    pythonw -m percept.percept

The above command will generate a pdf file with plots illustrating how the data was actively labeled (https://github.com/shubhomoydas/ad_examples/blob/master/documentation/percept_taurel_fixedtau_prior.pdf).

**Reference(s)**:
  - Das, S., Wong, W-K., Dietterich, T., Fern, A. and Emmott, A. (2016). Incorporating Expert Feedback into Active Anomaly Discovery in the Proceedings of the IEEE International Conference on Data Mining. (http://web.engr.oregonstate.edu/~wongwe/papers/pdf/ICDM2016.AAD.pdf)
  (https://github.com/shubhomoydas/aad/blob/master/overview/ICDM2016-AAD.pptx)

  - Das, S., Wong, W-K., Fern, A., Dietterich, T. and Siddiqui, A. (2017). Incorporating Feedback into Tree-based Anomaly Detection, KDD Interactive Data Exploration and Analytics (IDEA) Workshop.
  (http://poloclub.gatech.edu/idea2017/papers/p25-das.pdf)
  (https://github.com/shubhomoydas/pyaad/blob/master/presentations/IDEA17_slides.pptx)


Running AAD
-----------
This codebase supports four different anomaly detection algorithms:
  - The LODA based AAD (**works with streaming data, but does not support incremental update to model after building the model with the first window of data**)
  - The Isolation Forest based AAD (**streaming support with model update**)
    - For streaming update, we replace the oldest 20% trees with new trees trained on the latest window of data. The previously learned weights of the nodes of the retained (80%) trees are retained, and the weights of nodes of new trees are set to a default value (see code) before normalizing the entire weight vector to unit length.
  - HS Trees based AAD (**streaming support with model update**)
    - For streaming update, the option '--tree_update_type=0' replaces the previous node-level sample counts with counts from the new window of data. This is as per the original published algorithm. The option '--tree_update_type=1' updates the node-level counts as a linear combination of previous and current counts -- this is an experimental feature.
  - RS Forest based AAD (**streaming support with model update**)
    - See the previous HS Trees streaming update options above.

To run the Isolation Forest / HS-Trees / RS-Forest / LODA based algorithms, the command has the following format (**remember to run the commands from the 'python' folder, and monitor progress in logs under 'python/temp' folder**):

    bash ./aad.sh <dataset> <budget> <reruns> <tau> <detector_type> <query_type> <query_confident[0|1]> <streaming[0|1]> <streaming_window> <retention_type[0|1]> <with_prior[0|1]> <init_type[0|1|2]>

    for Isolation Forest, set <detector_type>=7; 
    for HSTrees, set <detector_type>=11;
    for RSForest, set <detector_type>=12;
    for LODA, set <detector_type>=13;

Example (with Isolation Forest, non-streaming):

    bash ./aad.sh toy2 35 1 0.03 7 1 0 0 512 0 1 1

Note: The above will generate 2D plots (tree partitions and score contours) under the 'temp' folder since <i>toy2</i> is a 2D dataset.

example (with HSTrees streaming):

    bash ./aad.sh toy2 35 1 0.03 11 1 0 1 256 0 1 1

**Note:** In case the data does not have concept drift, I would **recommend using Isolation forest** instead of HSTrees and RSForest:

    bash ./aad.sh toy2 35 1 0.03 7 1 0 1 512 1 1 1


**Note on Streaming**
Streaming currently supports two strategies for data retention:
  - Retention Type 0: Here the new instances from the stream completely overwrite the older *unlabeled instances* in memory.
  - Retention Type 1: Here the new instances are first merged with the older unlabeled instances and then the complete set is sorted in descending order on the distance from the margin. The top instances are retained; rest are discarded. **This is highly recommended.**


**Note on Query Diversity**
See further below for diversity based querying strategy. The query_type=8 option selects this. **To actually see benefits of this option, set batch size to greater than 1 (e.g., 3)**.


Generating compact descriptions with AAD
-------------------------------------------
ADD, when used with a forest-based detector such as Isolation Forest, can output a compact set of subspaces that contain all labeled anomalies. The idea is explained in https://github.com/shubhomoydas/ad_examples/blob/master/documentation/anomaly_description/anomaly_description.pdf. Following illustrations show the results of this approach.

To generate the below, use the command:
    
    bash ./aad.sh toy2 35 1 0.03 7 1 0 0 512 0 1 1

![Contours](figures/aad/contours.png)

![Descriptions](figures/aad/description.png)


Query diversity with compact descriptions
-------------------------------------------
Again, the idea for querying a diverse set of instances without significantly affecting the anomaly detection efficiency is explained in https://github.com/shubhomoydas/ad_examples/blob/master/documentation/anomaly_description/anomaly_description.pdf.

To generate the below, use the command:
    
    bash ./aad.sh toy2 10 1 0.03 7 1 0 0 512 0 1 1

![Query Diversity](figures/aad/query_diversity.png)


Does Query diversity with compact descriptions help?
-------------------------------------------
The below plots show that the above diversity strategy indeed helps.

To generate the below plots, perform the following steps (**remember to run the commands from the 'python' folder, and monitor progress in logs under 'python/temp' folder**):

    - set N_BATCH=1 in aad.sh and then run the command:
    
        bash ./aad.sh toy2 45 10 0.03 7 1 0 0 512 0 1 1
        
    - set N_BATCH=3 in aad.sh, and run the following commands:
    
        bash ./aad.sh toy2 45 10 0.03 7 1 0 0 512 0 1 1
        bash ./aad.sh toy2 45 10 0.03 7 2 0 0 512 0 1 1
        bash ./aad.sh toy2 45 10 0.03 7 8 0 0 512 0 1 1

    - Next, generate anomaly discovery curves:
        
        pythonw -m aad.plot_aad_results
        
    - Finally, generate class diversity plot:
    
        pythonw -m aad.plot_class_diversity

![Diversity Effect](figures/aad/diversity_effect.png)


Differences between Isolation Forest, HS Trees, RS Forest
-------------------------------------------
This [document](https://github.com/shubhomoydas/ad_examples/blob/master/documentation/anomaly_description/anomaly_description.pdf) explains why Isolation Forest is more effective in incorporating feedback at the leaf level. This is illustrated in the figure below.

![Tree Differences](figures/aad/tree_differences.png)


Running AAD with precomputed anomaly scores
-------------------------------------------
In case scores from anomaly detector ensembles are available in a CSV file, then AAD can be run with the following command.

    pythonw -m aad.precomputed_aad --startcol=2 --labelindex=1 --header --randseed=42 --dataset=toy --datafile=../datasets/toy.csv --scoresfile=../datasets/toy_scores.csv --querytype=1 --detector_type=14 --constrainttype=4 --sigma2=0.5 --budget=35 --tau=0.03 --Ca=1 --Cn=1 --Cx=1 --withprior --unifprior --init=1 --runtype=simple --log_file=./temp/precomputed_aad.log --debug

**Note: The detector_type is 14** for precomputed scores. The input file and scores should have the same format as in the example files (toy.csv, toy_scores.csv). Also, make sure the initialization is at uniform (**--init=1**) for good label efficiency (maximum reduction in false positives with minimum labeling effort). If the weights are initialized to zero or random, the results will be poor. *Ensembles enable us to get a good starting point for active learning in this case.*


Activity Modeling
-----------------
A simple application of word2vec for activity modeling can be found [here](python/timeseries/activity_word2vec.py). We try to infer relative sensor locations from sequence of sensor triggerings. The true [floor plan](http://ailab.wsu.edu/casas/hh/hh101/profile/page-6.html) and the inferred sensor locations (**for sensor ids starting with 'M' and 'MA'**) are shown below.

![Floor Plan](datasets/CASAS/floor_plans/HH101-sensormap.png)

![Relative Sensor Locations with Word2Vec](datasets/CASAS/floor_plans/activity_sensors_d15_original_tsne.png)

Please refer to the following paper and the [CASAS website](http://ailab.wsu.edu/casas/hh) for the setup:
    D. Cook, A. Crandall, B. Thomas, and N. Krishnan.
    CASAS: A smart home in a box. IEEE Computer, 46(7):62-69, 2013.


Note on Spectral Clustering by label diffusion
----------------------------------------------
Although the python code has the implementation, the last step requires non-metric MDS transform and the scikit-learn implementation is not as good as R. Hence, use the R code (R/manifold_learn.R) for generating the transformed output.

For details, refer to:
Supervised and Semi-supervised Approaches Based on Locally-Weighted Logistic Regression by Shubhomoy Das, Travis Moore, Weng-keen Wong, Simone Stumpf, Ian Oberst, Kevin Mcintosh, Margaret Burnett, Artificial Intelligence, 2013.


Some thoughts on Active Anomaly Discovery
-----------------------------------------
*Question: Why should active learning help in anomaly detection with ensembles?* Let us assume data is uniformly distributed on a 2D unit sphere (this is a setting commonly analysed in active learning theory literature). When we treat the ensemble scores as 'features', then most anomaly 'feature' vectors will be closer to the uniform unit vector (uniform unit vector has the same values for all 'features' where 'd' is the number of ensembles) than non-anomalies because anomaly detectors tend to assign higher scores to anomalies. This is another way of saying that the average of the anomaly scores would be a good representative of anomalousness (dot product of the transformed 'features' with the uniform weight vector). Seen another way, the hyper-plane perpendicular to the uniform weight vector and offset by cos(pi.tau) should a good prior for the separating hyper-plane between anomalies and nominals. The classification rule is: *sign(w.x - cos(pi.tau))* such that +1 is anomaly, -1 is nominal. On real-world data, the true hyper-plane is not exactly same as the uniform vector, but should be close (else the anomaly detectors forming the ensemble are poor). AAD is basically trying to find this true hyper-plane by solving a large-margin classification problem. The example 'percept.percept' illustrates this where we have true anomaly distribution (red points in the plots) at a slight angle from the uniform weights. With active learning, the true anomaly region on the unit sphere (centered around blue line) can be discovered in a more efficient manner if we set the uniform vector as a prior. Most current theory on active learning revolves around learning hyper-planes passing through the origin. This theory can be applied to ensemble-based anomaly detection by introducing the fixed cos(pi.tau) bias (the green line in the plots represents the learned hyperplane; the red line is perpendicular to it).

**Caution:** By design, the uniform weight vector is more closely aligned with the ensemble score vectors of **true anomalies** than with the ensemble score vectors of true nominals. However, this alignment cannot be guaranteed when the score vectors are normalized to unit length (such that they all lie on a unit sphere). Still, if the number of ensemble members is very high -- such as with IForest where leaf nodes represent the members -- then the normalization is more likely to preserve the intended alignment. This is probably due to some properties of high-dimensional geometry. The distribution of the angles between the normalized score vectors and the uniform weight vector can be checked with aad.test_hyperplane_angles. The plotted histograms show that true anomalies are usually closer to uniform vector (measured in angles) when IForest is used, and the optimal hyperplane (computed with a perceptron) has an acute angle with uniform vector. As a recommendation: the IForest leaf-based scores may be normalied, but LODA based scores should *not* be normalied to unit length because the number of LODA projections is smaller.

