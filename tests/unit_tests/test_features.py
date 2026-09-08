def test_model_features(model, transaction):
    pred = model.named_steps["preprocessor"].transform(transaction)
    assert pred.shape[1] == 30
