Python Deep Outlier/Model Selection
==================================================
For a given new dataset, model selection is to select the most suitable outlier detection model from the candidate models.

Supported models
~~~~~~~~~~~~~~~~~~~
FMMS - Factorization
Machine-Based Unsupervised Model Selection Method. in SMC. 2022.

Usages
~~~~~~~~~~~~~~
.. code-block:: python

    from deepod.model_selection.fmms import FMMS

    # Load pre-stored data.
    # p_train is the performance matrix which stores the model performance on historical datasets.
    # f_train is the feature matrix which stores the features of historical datasets.
    p_train, f_train = utils.get_data(FEATURE_FILE, TARGET_FILE)
    
    # f_test is the feature of the target dataset.
    # this is an optional input.
    # It is also possible to feed the original target dataset into the model
    f_test = np.array([1.227677502, 0.547297297, 0.25, 0.013513514, 0.234693979, 0.121621622, 8.222222222, 28.71360895, 9.512437578,
                      -0.52749456, 13.58201658, 0.817380952, 0.758373016, 0.843551587, 0.829662698, 0.61265873, 0.54952381, -2.106840516,
                      2.106840516, 2.890371758, 4.997212274, 15, 4, 18, 0, 148, 0, 0, 3, 0.446808511, -0.350354369, -0.558601676,
                      0, 0, 0, 5, 0.2, 5.387046595, 2.302843498, 0.330402533, 2.208985222, 8, 2.933333333, 2, 1.569146973, 44])

    # Instantiate the FMMS model. 
    model = FMMS()

    # Retrain the FMMS model.
    # The trained model can be saved in a specified path.
    model.fit(f_train, p_train, save_path='./data/fmms.pt')

    # use FMMS to recommend models.
    # It returns the top n model for new data f_test
    model.predict(f=f_test, topn=5, load_path='./data/fmms.pt')

    


