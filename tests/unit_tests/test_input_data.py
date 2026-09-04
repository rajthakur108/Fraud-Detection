def test_input_data(transaction,input_columns):
    assert list(transaction.keys()) == input_columns
