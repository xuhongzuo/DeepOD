Python Deep Outlier/Anomaly Detection (DeepOD)
==================================================

.. image:: https://github.com/xuhongzuo/DeepOD/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/xuhongzuo/DeepOD/actions/workflows/testing.yml
   :alt: testing2

.. image:: https://coveralls.io/repos/github/xuhongzuo/DeepOD/badge.svg?branch=main
    :target: https://coveralls.io/github/xuhongzuo/DeepOD?branch=main
    :alt: coveralls

.. image:: https://static.pepy.tech/personalized-badge/deepod?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads
   :target: https://pepy.tech/project/deepod
   :alt: downloads
   

``DeepOD`` is an open-source python library for Deep Learning-based `Outlier Detection <https://en.wikipedia.org/wiki/Anomaly_detection>`_
and `Anomaly Detection <https://en.wikipedia.org/wiki/Anomaly_detection>`_. ``DeepOD`` supports tabular anomaly detection and time-series anomaly detection.


DeepOD includes **23** deep outlier detection / anomaly detection algorithms (in unsupervised/weakly-supervised paradigm). More baseline algorithms will be included later.



**DeepOD is featured for**:

* **Unified APIs** across various algorithms.
* **SOTA models** includes reconstruction-, representation-learning-, and self-superivsed-based latest deep learning methods.
* **Comprehensive Testbed** that can be used to directly test different models on benchmark datasets (highly recommend for academic research).
* **Versatile** in different data types including tabular and time-series data (DeepOD will support other data types like images, graph, log, trace, etc. in the future, welcome PR :telescope:).
* **Diverse Network Structures** can be plugged into detection models, we now support LSTM, GRU, TCN, Conv, and Transformer for time-series data.  (welcome PR as well :sparkles:)


If you are interested in our project, we are pleased to have your stars and forks :thumbsup: :beers: .


Installation
~~~~~~~~~~~~~~
The DeepOD framework can be installed via:


.. code-block:: bash


    pip install deepod


install a developing version (strongly recommend)


.. code-block:: bash


    git clone https://github.com/xuhongzuo/DeepOD.git
    cd DeepOD
    pip install .


Usages
~~~~~~~~~~~~~~~~~


**Directly use detection models in DeepOD:**

DeepOD can be used in a few lines of code. This API style is the same with sklearn and PyOD.


.. code-block:: python


    # unsupervised methods
    from deepod.models.dsvdd import DeepSVDD
    clf = DeepSVDD()
    clf.fit(X_train, y=None)
    scores = clf.decision_function(X_test)

    # weakly-supervised methods
    from deepod.models.devnet import DevNet
    clf = DevNet()
    clf.fit(X_train, y=semi_y) # semi_y uses 1 for known anomalies, and 0 for unlabeled data
    scores = clf.decision_function(X_test)


**Testbed usage:**

Testbed contains the whole process of testing an anomaly detection model, including data loading, preprocessing, anomaly detection, and evaluation. 

Please refer to ``testbed/``

* ``testbed/testbed_unsupervised_ad.py`` is for testing unsupervised tabular anomaly detection models.
 
* ``testbed/testbed_unsupervised_tsad.py`` is for testing unsupervised time-series anomaly detection models.


Key arguments:

* ``--input_dir``: name of the folder that contains datasets (.csv, .npy)

* ``--dataset``: "FULL" represents testing all the files within the folder, or a list of dataset names using commas to split them (e.g., "10_cover*,20_letter*")

* ``--model``: anomaly detection model name

* ``--runs``: how many times running the detection model, finally report an average performance with standard deviation values


Example: 

1. Download `ADBench <https://github.com/Minqi824/ADBench/tree/main/adbench/datasets/>`_ datasets.
2. modify the ``dataset_root`` variable as the directory of the dataset.
3. ``input_dir`` is the sub-folder name of the ``dataset_root``, e.g., ``Classical`` or ``NLP_by_BERT``.  
4. use the following command in the bash

.. code-block:: bash

   python testbed_unsupervised_ad.py --model DIF --runs 5 --input_dir ADBench
   






Implemented Models
~~~~~~~~~~~~~~~~~~~

**Tabular Anomaly Detection models:**

.. csv-table:: 
 :header: "Model", "Venue", "Year", "Type", "Title"
 :widths: 4, 4, 4, 8, 20 

 Deep SVDD, ICML, 2018, unsupervised, Deep One-Class Classification  [#Ruff2018Deep]_
 REPEN, KDD, 2018, unsupervised, Learning Representations of Ultrahigh-dimensional Data for Random Distance-based Outlier Detection [#Pang2019Repen]_
 RDP, IJCAI, 2020, unsupervised, Unsupervised Representation Learning by Predicting Random Distances [#Wang2020RDP]_
 RCA, IJCAI, 2021, unsupervised, RCA: A Deep Collaborative Autoencoder Approach for Anomaly Detection [#Liu2021RCA]_
 GOAD, ICLR, 2020, unsupervised, Classification-Based Anomaly Detection for General Data [#Bergman2020GOAD]_
 NeuTraL, ICML, 2021, unsupervised, Neural Transformation Learning for Deep Anomaly Detection Beyond Images [#Qiu2021Neutral]_
 ICL, ICLR, 2022, unsupervised, Anomaly Detection for Tabular Data with Internal Contrastive Learning
 DIF, TKDE, 2023, unsupervised, Deep Isolation Forest for Anomaly Detection
 SLAD, ICML, 2023, unsupervised, Fascinating Supervisory Signals and Where to Find Them: Deep Anomaly Detection with Scale Learning
 DevNet, KDD, 2019, weakly-supervised, Deep Anomaly Detection with Deviation Networks
 PReNet, KDD, 2023, weakly-supervised, Deep Weakly-supervised Anomaly Detection
 Deep SAD, ICLR, 2020, weakly-supervised, Deep Semi-Supervised Anomaly Detection
 FeaWAD, TNNLS, 2021, weakly-supervised, Feature Encoding with AutoEncoders for Weakly-supervised Anomaly Detection


**Time-series Anomaly Detection models:**

.. csv-table:: 
 :header: "Model", "Venue", "Year", "Type", "Title"
 :widths: 4, 4, 4, 8, 20 

 AnomalyTransformer, ICLR, 2022, unsupervised, Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy
 TranAD, VLDB, 2022, unsupervised, TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data  
 COUTA, arXiv, 2022, unsupervised, Calibrated One-class Classification for Unsupervised Time Series Anomaly Detection
 USAD, KDD, 2020, unsupervised, USAD: UnSupervised Anomaly Detection on Multivariate Time Series  
 DIF, TKDE, 2023, unsupervised, Deep Isolation Forest for Anomaly Detection
 TcnED, TNNLS, 2021, unsupervised, An Evaluation of Anomaly Detection and Diagnosis in Multivariate Time Series
 Deep SVDD (TS), ICML, 2018, unsupervised, Deep One-Class Classification  
 DevNet (TS), KDD, 2019, weakly-supervised, Deep Anomaly Detection with Deviation Networks
 PReNet (TS), KDD, 2023, weakly-supervised, Deep Weakly-supervised Anomaly Detection
 Deep SAD (TS), ICLR, 2020, weakly-supervised, Deep Semi-Supervised Anomaly Detection

NOTE:

- For Deep SVDD, DevNet, PReNet, and DeepSAD, we employ network structures that can handle time-series data. These models' classes have a parameter named  ``network`` in these models, by changing it, you can use different networks.   

- We currently support 'TCN', 'GRU', 'LSTM', 'Transformer', 'ConvSeq', and 'DilatedConv'.   


Citation
~~~~~~~~~~~~~~~~~
If you use this library in your work, please cite this paper:

Hongzuo Xu, Guansong Pang, Yijie Wang and Yongjun Wang, "Deep Isolation Forest for Anomaly Detection," in IEEE Transactions on Knowledge and Data Engineering, doi: 10.1109/TKDE.2023.3270293.


You can also use the BibTex entry below for citation.

.. code-block:: bibtex

   @ARTICLE{xu2023deep,
      author={Xu, Hongzuo and Pang, Guansong and Wang, Yijie and Wang, Yongjun},
      journal={IEEE Transactions on Knowledge and Data Engineering}, 
      title={Deep Isolation Forest for Anomaly Detection}, 
      year={2023},
      volume={},
      number={},
      pages={1-14},
      doi={10.1109/TKDE.2023.3270293}
   }



Reference
~~~~~~~~~~~~~~~~~

.. [#Ruff2018Deep] Ruff, Lukas, et al. "Deep one-class classification." ICML. 2018.

.. [#Pang2019Repen] Pang, Guansong, et al. "Learning representations of ultrahigh-dimensional data for random distance-based outlier detection". KDD (pp. 2041-2050). 2018.

.. [#Wang2020RDP] Wang, Hu, et al. "Unsupervised Representation Learning by Predicting Random Distances". IJCAI (pp. 2950-2956). 2020.

.. [#Liu2021RCA] Liu, Boyang, et al. "RCA: A Deep Collaborative Autoencoder Approach for Anomaly Detection". IJCAI (pp. 1505-1511). 2021.

.. [#Bergman2020GOAD] Bergman, Liron, and Yedid Hoshen. "Classification-Based Anomaly Detection for General Data". ICLR. 2020.

.. [#Qiu2021Neutral] Qiu, Chen, et al. "Neural Transformation Learning for Deep Anomaly Detection Beyond Images". ICML. 2021.
