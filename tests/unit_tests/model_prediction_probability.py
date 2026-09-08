def test_model_pred_probability(model,transaction):
    pred_proba = model.predict_proba(transaction)

    assert pred_proba.shape[1] == 2
    assert pred_proba[0][1]>= 0
    assert pred_proba[0][1]<= 1