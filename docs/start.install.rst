Installation
============

It is recommended to use **pip** for installation. Please make sure
**the latest version** is installed, as DeepOD is updated frequently:

.. code-block:: bash

   pip install deepod            # normal install
   pip install --upgrade deepod  # or update if needed


Alternatively, you could clone and run setup.py file:

.. code-block:: bash

   git clone https://github.com/xuhongzuo/deepod.git
   cd pyod
   pip install .


**Required Dependencies**\ :


* Python 3.7+
* numpy>=1.19
* scipy>=1.5.1
* scikit_learn>=0.20.0
* pandas>=1.0.0
* torch>1.10.0,<1.13.1
* ray==2.6.1
* pyarrow>=11.0.0
* einops

