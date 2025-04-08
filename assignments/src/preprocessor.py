from collections import Counter
import numpy as np
import pandas as pd

def Reweighing(X, Y, A):
    # X: independent variables (2-d pd.DataFrame)
    # Y: the dependent variable (1-d np.array, binary y in {0,1})
    # A: a list/array of the names of the sensitive attributes with binary values
    # Return: sample_weight, an array of float weight for every data point
    #         sample_weight(a,y) = P(y)*P(a)/P(a,y)
    # Write your code below:

    X = pd.DataFrame(X)
    data = X[A].copy()
    data['Y'] = Y

    total = len(data)
    p_y = data['Y'].value_counts(normalize=True).to_dict()
    p_a = data[A].value_counts(normalize=True).to_dict()
    p_ay = data.groupby(A + ['Y']).size() / total

    sample_weight = []
    for _, row in data.iterrows():
        a_key = tuple(row[a] for a in A)
        y_val = row['Y']
        try:
            weight = (p_y[y_val] * p_a[a_key]) / p_ay.loc[(*a_key, y_val)]
        except KeyError:
            weight = 0
        sample_weight.append(weight)

    sample_weight = np.array(sample_weight)


    # Rescale the sum of sample weights to len(y) before returning it
    sample_weight = sample_weight * len(Y) / sum(sample_weight)
    return sample_weight


