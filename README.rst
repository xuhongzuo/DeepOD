Python Deep Outlier/Anomaly Detection (DeepOD)
==================================================

.. image:: https://github.com/xuhongzuo/DeepOD/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/xuhongzuo/DeepOD/actions/workflows/testing.yml
   :alt: testing2

.. image:: https://readthedocs.org/projects/deepod/badge/?version=latest
    :target: https://deepod.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://app.codacy.com/project/badge/Grade/2c587126aac2441abb917c032189fbe8
    :target: https://app.codacy.com/gh/xuhongzuo/DeepOD/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade
    :alt: codacy

.. image:: https://coveralls.io/repos/github/xuhongzuo/DeepOD/badge.svg?branch=main
    :target: https://coveralls.io/github/xuhongzuo/DeepOD?branch=main
    :alt: coveralls

.. image:: https://static.pepy.tech/personalized-badge/deepod?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads
   :target: https://pepy.tech/project/deepod
   :alt: downloads

.. image:: https://img.shields.io/badge/license-BSD2-blue
   :alt: license

   

``DeepOD`` is an open-source python library for Deep Learning-based `Outlier Detection <https://en.wikipedia.org/wiki/Anomaly_detection>`_
and `Anomaly Detection <https://en.wikipedia.org/wiki/Anomaly_detection>`_. ``DeepOD`` supports tabular anomaly detection and time-series anomaly detection.


DeepOD includes **27** deep outlier detection / anomaly detection algorithms (in unsupervised/weakly-supervised paradigm).
More baseline algorithms will be included later.



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


Directly use detection models in DeepOD:
::::::::::::::::::::::::::::::::::::::::::

DeepOD can be used in a few lines of code. This API style is the same with `Sklean <https://github.com/scikit-learn/scikit-learn>`_ and `PyOD <https://github.com/yzhao062/pyod>`_.


**for tabular anomaly detection:**

.. code-block:: python


    # unsupervised methods
    from deepod.models.tabular import DeepSVDD
    clf = DeepSVDD()
    clf.fit(X_train, y=None)
    scores = clf.decision_function(X_test)

    # weakly-supervised methods
    from deepod.models.tabular import DevNet
    clf = DevNet()
    clf.fit(X_train, y=semi_y) # semi_y uses 1 for known anomalies, and 0 for unlabeled data
    scores = clf.decision_function(X_test)

    # evaluation of tabular anomaly detection
    from deepod.metrics import tabular_metrics
    auc, ap, f1 = tabular_metrics(y_test, scores)


**for time series anomaly detection:**


.. code-block:: python


    # time series anomaly detection methods
    from deepod.models.time_series import TimesNet
    clf = TimesNet()
    clf.fit(X_train)
    scores = clf.decision_function(X_test)

    # evaluation of time series anomaly detection
    from deepod.metrics import ts_metrics
    from deepod.metrics import point_adjustment # execute point adjustment for time series ad
    eval_metrics = ts_metrics(labels, scores)
    adj_eval_metrics = ts_metrics(labels, point_adjustment(labels, scores))
    




Testbed usage:
::::::::::::::::::::::::::::::::::::::::::


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

    
    cd DeepOD
    pip install .
    cd testbed
    python testbed_unsupervised_ad.py --model DeepIsolationForest --runs 5 --input_dir ADBench
   



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
 ICL, ICLR, 2022, unsupervised, Anomaly Detection for Tabular Data with Internal Contrastive Learning [#Shenkar2022ICL]_
 DIF, TKDE, 2023, unsupervised, Deep Isolation Forest for Anomaly Detection [#Xu2023DIF]_
 SLAD, ICML, 2023, unsupervised, Fascinating Supervisory Signals and Where to Find Them: Deep Anomaly Detection with Scale Learning [#Xu2023SLAD]_
 DevNet, KDD, 2019, weakly-supervised, Deep Anomaly Detection with Deviation Networks [#Pang2019DevNet]_
 PReNet, KDD, 2023, weakly-supervised, Deep Weakly-supervised Anomaly Detection [#Pang2023PreNet]_
 Deep SAD, ICLR, 2020, weakly-supervised, Deep Semi-Supervised Anomaly Detection [#Ruff2020DSAD]_
 FeaWAD, TNNLS, 2021, weakly-supervised, Feature Encoding with AutoEncoders for Weakly-supervised Anomaly Detection [#Zhou2021FeaWAD]_
 RoSAS, IP&M, 2023, weakly-supervised, RoSAS: Deep semi-supervised anomaly detection with contamination-resilient continuous supervision [#Xu2023RoSAS]_

**Time-series Anomaly Detection models:**

.. csv-table:: 
 :header: "Model", "Venue", "Year", "Type", "Title"
 :widths: 4, 4, 4, 8, 20 

 DCdetector, KDD, 2023, unsupervised, DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection [#Yang2023dcdetector]_
 TimesNet, ICLR, 2023, unsupervised, TIMESNET: Temporal 2D-Variation Modeling for General Time Series Analysis [#Wu2023timesnet]_
 AnomalyTransformer, ICLR, 2022, unsupervised, Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy [#Xu2022transformer]_
 NCAD, IJCAI, 2022, unsupervised, Neural Contextual Anomaly Detection for Time Series [#Carmona2022NCAD]_
 TranAD, VLDB, 2022, unsupervised, TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data [#Tuli2022TranAD]_
 COUTA, arXiv, 2022, unsupervised, Calibrated One-class Classification for Unsupervised Time Series Anomaly Detection [#Xu2022COUTA]_
 USAD, KDD, 2020, unsupervised, USAD: UnSupervised Anomaly Detection on Multivariate Time Series  
 DIF, TKDE, 2023, unsupervised, Deep Isolation Forest for Anomaly Detection [#Xu2023DIF]_
 TcnED, TNNLS, 2021, unsupervised, An Evaluation of Anomaly Detection and Diagnosis in Multivariate Time Series [#Garg2021Evaluation]_
 Deep SVDD (TS), ICML, 2018, unsupervised, Deep One-Class Classification [#Ruff2018Deep]_
 DevNet (TS), KDD, 2019, weakly-supervised, Deep Anomaly Detection with Deviation Networks [#Pang2019DevNet]_
 PReNet (TS), KDD, 2023, weakly-supervised, Deep Weakly-supervised Anomaly Detection [#Pang2023PreNet]_
 Deep SAD (TS), ICLR, 2020, weakly-supervised, Deep Semi-Supervised Anomaly Detection [#Ruff2020DSAD]_

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


Star History
~~~~~~~~~~~~~~~~~

Current stars:

.. image:: https://img.shields.io/github/stars/xuhongzuo/deepod?labelColor=black&color=red
   :alt: GitHub Repo stars


.. image:: https://api.star-history.com/svg?repos=xuhongzuo/DeepOD&type=Date
   :target: https://star-history.com/#xuhongzuo/DeepOD&Date
   :align: center





Reference
~~~~~~~~~~~~~~~~~

.. [#Ruff2018Deep] Ruff, Lukas, et al. "Deep one-class classification." ICML. 2018.

.. [#Pang2019Repen] Pang, Guansong, et al. "Learning representations of ultrahigh-dimensional data for random distance-based outlier detection". KDD (pp. 2041-2050). 2018.

.. [#Wang2020RDP] Wang, Hu, et al. "Unsupervised Representation Learning by Predicting Random Distances". IJCAI (pp. 2950-2956). 2020.

.. [#Liu2021RCA] Liu, Boyang, et al. "RCA: A Deep Collaborative Autoencoder Approach for Anomaly Detection". IJCAI (pp. 1505-1511). 2021.

.. [#Bergman2020GOAD] Bergman, Liron, and Yedid Hoshen. "Classification-Based Anomaly Detection for General Data". ICLR. 2020.

.. [#Qiu2021Neutral] Qiu, Chen, et al. "Neural Transformation Learning for Deep Anomaly Detection Beyond Images". ICML. 2021.

.. [#Shenkar2022ICL] Shenkar, Tom, et al. "Anomaly Detection for Tabular Data with Internal Contrastive Learning". ICLR. 2022.

.. [#Pang2019DevNet] Pang, Guansong, et al. "Deep Anomaly Detection with Deviation Networks". KDD. 2019.

.. [#Pang2023PreNet] Pang, Guansong, et al. "Deep Weakly-supervised Anomaly Detection". KDD. 2023. 

.. [#Ruff2020DSAD] Ruff, Lukas, et al. "Deep Semi-Supervised Anomaly Detection". ICLR. 2020. 

.. [#Zhou2021FeaWAD] Zhou, Yingjie, et al. "Feature Encoding with AutoEncoders for Weakly-supervised Anomaly Detection". TNNLS. 2021. 

.. [#Xu2022transformer] Xu, Jiehui, et al. "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy". ICLR, 2022.

.. [#Wu2023timesnet] Wu, Haixu, et al. "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis". ICLR. 2023.

.. [#Yang2023dcdetector] Yang, Yiyuan, et al. "DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection". KDD. 2023

.. [#Tuli2022TranAD] Tuli, Shreshth, et al. "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data". VLDB. 2022.

.. [#Carmona2022NCAD] Carmona, Chris U., et al. "Neural Contextual Anomaly Detection for Time Series". IJCAI. 2022. 

.. [#Garg2021Evaluation] Garg, Astha, et al. "An Evaluation of Anomaly Detection and Diagnosis in Multivariate Time Series". TNNLS. 2021. 

.. [#Xu2022COUTA] Xu, Hongzuo et al. "Calibrated One-class Classification for Unsupervised Time Series Anomaly Detection". arXiv:2207.12201. 2022.

.. [#Xu2023DIF] Xu, Hongzuo et al. "Deep Isolation Forest for Anomaly Detection". TKDE. 2023.

.. [#Xu2023SLAD] Xu, Hongzuo et al. "Fascinating supervisory signals and where to find them: deep anomaly detection with scale learning". ICML. 2023. 

.. [#Xu2023RoSAS] Xu, Hongzuo et al. "RoSAS: Deep semi-supervised anomaly detection with contamination-resilient continuous supervision". IP&M. 2023. 


