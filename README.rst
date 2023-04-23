Python Deep Outlier/Anomaly Detection (DeepOD)
==================================================

.. image:: https://github.com/xuhongzuo/DeepOD/actions/workflows/testing_conda.yml/badge.svg
   :target: https://github.com/xuhongzuo/DeepOD/actions/workflows/testing_conda.yml
   :alt: testing

.. image:: https://github.com/xuhongzuo/DeepOD/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/xuhongzuo/DeepOD/actions/workflows/testing.yml
   :alt: testing2

.. image:: https://coveralls.io/repos/github/xuhongzuo/DeepOD/badge.svg?branch=main
    :target: https://coveralls.io/github/xuhongzuo/DeepOD?branch=main
    :alt: coveralls

.. image:: https://static.pepy.tech/personalized-badge/deepod?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads
   :target: https://pepy.tech/project/deepod
   :alt: downloads
   

**DeepOD** is an open-source python framework for deep learning-based anomaly detection on multivariate/time-series data. DeepOD provides unified implementation of different detection models based on PyTorch.


DeepOD includes twelve deep outlier detection / anomaly detection algorithms (in unsupervised/weakly-supervised paradigm) for now. More baseline algorithms will be included later.


🔭 *We are working on a new feature -- by simply setting a few parameters, different deep anomaly detection models can not only handle different data types.*   

- We have finished some attempts on partial models like Deep SVDD, DevNet, Deep SAD, PReNet and DIF. These models can use temporal networks like LSTM, GRU, TCN, Conv, Transformer to handle time series data. 
- *Future work*: we also want to implement several network structure, so as to processing more data types like graphs and images by simply plugging in corresponding network architecture. 


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


Supported Models
~~~~~~~~~~~~~~~~~

**Detection models:**

.. csv-table:: 
 :header: "Model", "Venue", "Year", "Type", "Title"
 :widths: 4, 4, 4, 8, 20 

 Deep SVDD, ICML, 2018, unsupervised, Deep One-Class Classification  
 REPEN, KDD, 2018, unsupervised, Learning Representations of Ultrahigh-dimensional Data for Random Distance-based Outlier Detection
 RDP, IJCAI, 2020, unsupervised, Unsupervised Representation Learning by Predicting Random Distances  
 RCA, IJCAI, 2021, unsupervised, RCA: A Deep Collaborative Autoencoder Approach for Anomaly Detection
 GOAD, ICLR, 2020, unsupervised, Classification-Based Anomaly Detection for General Data
 NeuTraL, ICML, 2021, unsupervised, Neural Transformation Learning for Deep Anomaly Detection Beyond Images
 ICL, ICLR, 2022, unsupervised, Anomaly Detection for Tabular Data with Internal Contrastive Learning
 DIF, TKDE, 2023, unsupervised, Deep Isolation Forest for Anomaly Detection
 DevNet, KDD, 2019, weakly-supervised, Deep Anomaly Detection with Deviation Networks
 PReNet, ArXiv, 2020, weakly-supervised, Deep Weakly-supervised Anomaly Detection
 Deep SAD, ICLR, 2020, weakly-supervised, Deep Semi-Supervised Anomaly Detection
 FeaWAD, TNNLS, 2021, weakly-supervised, Feature Encoding with AutoEncoders for Weakly-supervised Anomaly Detection


Usages
~~~~~~~~~~~~~~~~~


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



Citation
~~~~~~~~~~~~~~~~~
If you use this library in your work, please use the BibTex entry below for citation.

.. code-block:: bibtex

   @misc{deepod,
      author = {{Xu, Hongzuo}},
      title = {{DeepOD: Python Deep Outlier/Anomaly Detection}},
      url = {https://github.com/xuhongzuo/DeepOD},
      version = {0.2},
   }
