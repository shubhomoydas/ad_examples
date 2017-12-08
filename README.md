Anomaly Detection Examples
--------------------------
This is a collection of anomaly detection examples for detection methods popular in academic literature and in practice. I will include more examples as and when I find time.

To execute the code:

1. Run code from 'python' folder. The outputs will be generated under 'temp' folder. The 'pythonw' command is used on OSX, but 'python' should be used on Linux.

2. The run commands are at the top of the python source code files.


Python libraries required:
--------------------------
    numpy
    scipy
    scikit-learn (0.19.1)
    pandas
    ranking
    statsmodels
    matplotlib


Active Anomaly Discovery
------------------------
The 'pyaad' project (https://github.com/shubhomoydas/pyaad) implements an algorithm (AAD) to actively explore anomalies. **Assuming that the ensemble scores have already been computed**, the file (https://github.com/shubhomoydas/ad_examples/blob/master/python/percept/percept.py) implements AAD in a much more simplified manner. The main observation is that we can normalize all the (transformed) vectors such that they lie on the surface of a unit sphere. With this, the top 'tau'-th quantile score can be assumed to be fixed at (1-tau) under uniform distribution assumption.

To run (https://github.com/shubhomoydas/ad_examples/blob/master/python/percept/percept.py):

    pythonw -m percept.percept

The above command will generate a pdf file with plots illustrating how the data was actively labeled (https://github.com/shubhomoydas/ad_examples/blob/master/documentation/percept_taurel_fixedtau_prior.pdf).

*Question: Why should active learning help in anomaly detection with ensembles?* When we treat the ensemble scores as 'features', then most anomaly 'feature' vectors will be closer to the uniform unit vector (uniform vector has the same values for all 'features' where 'd' is the number of ensembles) than non-anomalies. This is another way of saying that the average of the anomaly scores would be a good representative of anomalousness (dot product of the transformed 'features' with the uniform weight vector). Seen another way, the hyper-plane perpendicular to the uniform weight vector and offset by (1-tau) should a good prior for the separating hyper-plane between anomalies and nominals. The classification rule is: *sign(w.x - (1-tau))* such that +1 is anomaly, -1 is nominal. On real-world data, the true hyper-plane is not exactly same as the uniform vector, but should be close (else the anomaly detectors forming the ensemble are poor). AAD is basically trying to find this true hyper-plane by solving a large-margin classification problem. The example 'percept.percept' illustrates this where we have true anomaly distribution (red points in the plots) at a slight angle from the uniform weights. With active learning, the true anomaly region on the unit sphere (centered around blue line) can be discovered in a more efficient manner if we set the uniform vector as a prior. Most current theory on active learning revolves around learning hyper-planes passing through the origin. This theory can be applied to ensemble-based anomaly detection by introducing the fixed (1-tau) bias (green line in the plots).


Note on Spectral Clustering by label diffusion
----------------------------------------------
Although the python code has the implementation, the last step requires non-metric MDS transform and the scikit-learn implementation is not as good as R. Hence, use the R code (R/manifold_learn.R) for generating the transformed output.

For details, refer to:
Supervised and Semi-supervised Approaches Based on Locally-Weighted Logistic Regression by Shubhomoy Das, Travis Moore, Weng-keen Wong, Simone Stumpf, Ian Oberst, Kevin Mcintosh, Margaret Burnett, Artificial Intelligence, 2013.
