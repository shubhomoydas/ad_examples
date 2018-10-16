Activity Modeling
-----------------
A simple application of word2vec for activity modeling can be found [here](python/timeseries/activity_word2vec.py). We try to infer relative sensor locations from sequence of sensor triggerings. The true [floor plan](http://ailab.wsu.edu/casas/hh/hh101/profile/page-6.html) and the inferred sensor locations (**for sensor ids starting with 'M' and 'MA'**) are shown below ([download the data here](http://casas.wsu.edu/datasets/hh101.zip)). This demonstrates a form of 'embedding' of the sensors in a latent space. The premise is that the **non-iid data such as activity sequences may be represented in the latent space as i.i.d data on which standard anomaly detectors may be employed**. We can be a bit more creative and try to apply **transfer learning** with this embedding.

For example, imagine that we have a house (House-1) with labeled sensors (such as 'kitchen', 'living room', etc.) and another (House-2) with partially labeled sensors. Then, if we try to reduce the 'distance' between similarly labeled sensors in the latent space (by adding another loss-component to the word2vec embeddings), it can provide more information on which of the unlabeled sensors and activities in House-2 are similar to those in House-1. Moreover, the latent space allows representation of heterogeneous entities such as sensors, activities, locations, etc. in the same space which (in theory) helps detect similarities and associations in a more straightforward manner. In practice, the amount of data and the quality of the loss function matter a lot. Moreover, simpler methods of finding similarities/associations should not be overlooked. As an example, we might try to use embedding to figure out if a particular sensor is located in the bedroom. However, it might be simpler to just use the sensor's activation time to determine this information (assuming people sleep regular hours).

![Floor Plan](datasets/CASAS/floor_plans/HH101-sensormap.png)

![Relative Sensor Locations with Word2Vec](datasets/CASAS/floor_plans/activity_sensors_d15_original_tsne.png)

Please refer to the following paper and the [CASAS website](http://ailab.wsu.edu/casas/hh) for the setup:
    D. Cook, A. Crandall, B. Thomas, and N. Krishnan.
    CASAS: A smart home in a box. IEEE Computer, 46(7):62-69, 2013.


