Model Save & Load 
==================

The detection model class has ``save_model`` and ``load_model`` functions. 

We take the `DeepSVDD` model for example. 

.. code-block:: python
    
    from deepod.models import DeepSVDD

    # training an anomaly detection model
    model = DeepSVDD() # or any other models in DeepOD
    model.fit(X_train) # training

    path = 'save_file.pkl'
    model.save_model(path) # save trained model at the assigned path

    # directly load trained model from path
    model = DeepSVDD.load_model(path)
    model.decision_function(X_test)
    # or
    model.predict(X_test)



You can also directly use pickle for saving and loading DeepOD models. 

.. code-block:: python
    
    import pickle
    from deepod.models import DeepSVDD

    model = DeepSVDD()
    model.fit(X_train)

    with open('save_file.pkl', 'wb'):
        pickle.dump(model)

    with open('save_file.pkl', 'rb')
        model = pickle.load(f)

    model.decision_function(X_test)


