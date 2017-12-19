Anomaly Detection Examples
--------------------------
This is a collection of anomaly detection examples for detection methods popular in academic literature and in practice. I will include more examples as and when I find time.

To execute the code:

1. Run code from 'python' folder. The outputs will be generated under 'temp' folder. The 'pythonw' command is used on OSX, but 'python' should be used on Linux.

2. The run commands are at the top of the python source code files.


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


Active Anomaly Discovery (AAD)
------------------------------
The 'pyaad' project (https://github.com/shubhomoydas/pyaad) implements an algorithm (AAD) to actively explore anomalies. **Assuming that the ensemble scores have already been computed**, the file (https://github.com/shubhomoydas/ad_examples/blob/master/python/percept/percept.py) implements AAD in a much more simplified manner.

To run (https://github.com/shubhomoydas/ad_examples/blob/master/python/percept/percept.py):

    pythonw -m percept.percept

The above command will generate a pdf file with plots illustrating how the data was actively labeled (https://github.com/shubhomoydas/ad_examples/blob/master/documentation/percept_taurel_fixedtau_prior.pdf).

*Question: Why should active learning help in anomaly detection with ensembles?* Let us assume data is uniformly distributed on a 2D unit sphere (this is a setting commonly analysed in active learning theory literature). When we treat the ensemble scores as 'features', then most anomaly 'feature' vectors will be closer to the uniform unit vector (uniform unit vector has the same values for all 'features' where 'd' is the number of ensembles) than non-anomalies because anomaly detectors tend to assign higher scores to anomalies. This is another way of saying that the average of the anomaly scores would be a good representative of anomalousness (dot product of the transformed 'features' with the uniform weight vector). Seen another way, the hyper-plane perpendicular to the uniform weight vector and offset by cos(pi.tau) should a good prior for the separating hyper-plane between anomalies and nominals. The classification rule is: *sign(w.x - cos(pi.tau))* such that +1 is anomaly, -1 is nominal. On real-world data, the true hyper-plane is not exactly same as the uniform vector, but should be close (else the anomaly detectors forming the ensemble are poor). AAD is basically trying to find this true hyper-plane by solving a large-margin classification problem. The example 'percept.percept' illustrates this where we have true anomaly distribution (red points in the plots) at a slight angle from the uniform weights. With active learning, the true anomaly region on the unit sphere (centered around blue line) can be discovered in a more efficient manner if we set the uniform vector as a prior. Most current theory on active learning revolves around learning hyper-planes passing through the origin. This theory can be applied to ensemble-based anomaly detection by introducing the fixed cos(pi.tau) bias (the green line in the plots represents the learned hyperplane; the red line is perpendicular to it).

**Caution: If you are normalizing the scores to unit length such that your data lies on a unit sphere, then the alignment with uniform vector will hold true if the number of ensemble members is very high -- like with IForest where leaf nodes represent the members. I think this is a property of high-dimensional geometry.** You can check out the distribution of angles of instances with the uniform weight vector using aad.test_hyperplane_angles and aad.loda. The true anomalies are usually closer to uniform vector when IForest is used, and the optimal hyperplane (computed with a perceptron) has an acute angle with uniform vector.


##Reference(s):
  - Das, S., Wong, W-K., Dietterich, T., Fern, A. and Emmott, A. (2016). Incorporating Expert Feedback into Active Anomaly Discovery in the Proceedings of the IEEE International Conference on Data Mining. (http://web.engr.oregonstate.edu/~wongwe/papers/pdf/ICDM2016.AAD.pdf)
  (https://github.com/shubhomoydas/aad/blob/master/overview/ICDM2016-AAD.pptx)

  - Das, S., Wong, W-K., Fern, A., Dietterich, T. and Siddiqui, A. (2017). Incorporating Feedback into Tree-based Anomaly Detection, KDD Interactive Data Exploration and Analytics (IDEA) Workshop.
  (http://poloclub.gatech.edu/idea2017/papers/p25-das.pdf)
  (https://github.com/shubhomoydas/pyaad/blob/master/presentations/IDEA17_slides.pptx)


Running the tree-based AAD
--------------------------
This codebase has three different algorithms:
  - The LODA based AAD (**supports streaming, but not incremental update to model**)
  - The Isolation Forest based AAD (**supports streaming, but not incremental update to model**)
  - HS Trees based AAD (streaming support with model update)
  - RS Forest based AAD (streaming support with model update)

To run the Isolation Forest / HS-Trees / RS-Forest based algorithms, the command has the following format:

    bash ./aad.sh <dataset> <budget> <reruns> <tau> <detector_type> <query_type> <query_confident[0|1]> <streaming[0|1]> <streaming_window> <retention_type[0|1]>

    for Isolation Forest, set <detector_type>=7; 
    for HSTrees, set <detector_type>=11;
    for RSForest, set <detector_type>=12;
    for LODA, set <detector_type>=13;

example (with Isolation Forest, non-streaming):

    bash ./aad.sh toy2 35 1 0.03 7 1 0 0 512 0

Note: The above will generate 2D plots (tree partitions and score contours) under the 'temp' folder since <i>toy2</i> is a 2D dataset.

example (with HSTrees streaming):

    bash ./aad.sh toy2 35 1 0.03 11 1 0 1 256 0

**Note:** In case the data does not have concept drift, I would **recommend using Isolation forest** instead of HSTrees and RSForest:

    bash ./aad.sh toy2 35 1 0.03 7 1 0 1 512 1


# Note on Streaming
Streaming currently supports two strategies for data retention:
  - Retention Type 0: Here the new instances from the stream completely overwrite the older *unlabeled instances* in memory.
  - Retention Type 1: Here the new instances are first merged with the older unlabeled instances and then the complete set is sorted in descending order on the distance from the margin. The top instances are retained; rest are discarded. **This is highly recommended.**


Note on Spectral Clustering by label diffusion
----------------------------------------------
Although the python code has the implementation, the last step requires non-metric MDS transform and the scikit-learn implementation is not as good as R. Hence, use the R code (R/manifold_learn.R) for generating the transformed output.

For details, refer to:
Supervised and Semi-supervised Approaches Based on Locally-Weighted Logistic Regression by Shubhomoy Das, Travis Moore, Weng-keen Wong, Simone Stumpf, Ian Oberst, Kevin Mcintosh, Margaret Burnett, Artificial Intelligence, 2013.
