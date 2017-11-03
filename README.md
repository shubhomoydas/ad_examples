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



Note on Spectral Clustering by label diffusion
----------------------------------------------
Although the python code has the implementation, the last step requires non-metric MDS transform and the scikit-learn implementation is not as good as R. Hence, use the R code (R/manifold_learn.R) for generating the transformed output.

For details, refer to:
Supervised and Semi-supervised Approaches Based on Locally-Weighted Logistic Regression by Shubhomoy Das, Travis Moore, Weng-keen Wong, Simone Stumpf, Ian Oberst, Kevin Mcintosh, Margaret Burnett, Artificial Intelligence, 2013.
