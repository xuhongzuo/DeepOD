
.. DeepOD documentation master file, created by
   sphinx-quickstart on Tue Nov  7 21:28:52 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to DeepOD documentation!
==================================


.. image:: https://github.com/xuhongzuo/DeepOD/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/xuhongzuo/DeepOD/actions/workflows/testing.yml
   :alt: testing2

.. image:: https://readthedocs.org/projects/deepod/badge/?version=latest
    :target: https://deepod.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/xuhongzuo/DeepOD/badge.svg?branch=main
    :target: https://coveralls.io/github/xuhongzuo/DeepOD?branch=main
    :alt: coveralls

.. image:: https://static.pepy.tech/personalized-badge/deepod?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads
   :target: https://pepy.tech/project/deepod
   :alt: downloads




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




----


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   start.install
   start.examples
   start.model_save

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   api_reference
   api_cc

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   additional.contributing
   additional.license
   additional.star_history