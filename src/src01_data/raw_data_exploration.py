import os
import sys
import numpy as np
import pandas as pd

from src.src00_utils import data_cleaning

# test function
print(data_cleaning.encode_units(5))
print(data_cleaning.encode_units(1))
print(data_cleaning.encode_units(0))

pd.read