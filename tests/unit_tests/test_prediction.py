def test_model_prediction(model,transaction):
    
    prediction = model.predict(transaction)

    assert int(prediction[0]) in [0,1]