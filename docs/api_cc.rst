API CheatSheet
==============

The following APIs are applicable for all detector models for easy use.

* :func:`deepod.core.base_model.BaseDeepAD.fit`: Fit detector. y is ignored in unsupervised methods.
* :func:`deepod.core.base_model.BaseDeepAD.decision_function`: Predict raw anomaly score of X using the fitted detector.
* :func:`deepod.core.base_model.BaseDeepAD.predict`: Predict if a particular sample is an outlier or not using the fitted detector.


Key Attributes of a fitted model:

* :attr:`deepod.core.base_model.BaseDeepAD.decision_scores_`: The outlier scores of the training data. The higher, the more abnormal.
  Outliers tend to have higher scores.
* :attr:`deepod.core.base_model.BaseDeepAD.labels_`: The binary labels of the training data. 0 stands for inliers and 1 for outliers/anomalies.


See base class definition below:

deepod.core.base_model module
-----------------------

.. automodule:: deepod.core.base_model
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

