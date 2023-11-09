Examples
=========


Directly Use Detection Models
------------------------------


DeepOD can be used in a few lines of code. 
This API style is the same with `Sklean <https://github.com/scikit-learn/scikit-learn>`_ and `PyOD <https://github.com/yzhao062/pyod>`_.


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
    


Testbed
--------



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
   
