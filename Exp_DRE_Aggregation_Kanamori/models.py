

def nmse(y_true, y_pred):
    return ((y_true / (y_true.mean() + 1e-5) - y_pred / (y_pred.mean() + 1e-5)) ** 2).mean()



