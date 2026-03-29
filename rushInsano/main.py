from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd

df = pd.read_csv()
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
