Differences between Isolation Forest, HS Trees, RS Forest
---------------------------------------------------------
This [document](https://github.com/shubhomoydas/ad_examples/blob/master/documentation/anomaly_description/anomaly_description.pdf) explains why Isolation Forest is more effective in incorporating feedback at the leaf level. This is illustrated in the figure below. The plots are generated in the files `query_candidate_regions_ntop5_*.pdf` and `query_compact_ntop5_*.pdf` under `temp/aad/toy2/*` when the following commands are executed:

    bash ./aad.sh toy2 35 1 0.03 7 1 0 0 512 0 1 1
    bash ./aad.sh toy2 35 1 0.03 11 1 0 0 512 0 1 1
    bash ./aad.sh toy2 35 1 0.03 12 1 0 0 512 0 1 1

![Tree Differences](figures/aad/tree_differences.png)

