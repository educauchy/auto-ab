import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
import math, os
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from scipy.stats import mannwhitneyu, ttest_ind, kstwo
from typing import Dict, List, Tuple, Any, Union, Optional
from collections.abc import Callable


class Splitter:
    def __init__(self) -> None:
        pass

    def set_splitter(self, splitter: Callable[[], [np.array, np.array]]):
        pass

    def split(self, use_stratification: bool = True):
        pass