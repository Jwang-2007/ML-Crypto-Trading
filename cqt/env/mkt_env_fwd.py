import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

from cqt.error_msg import error
from cqt.env.mkt_env import MktEnvSec


class MktEnvrFwd(MktEnvSec):
    def __init__(self, target, data_collection, config=None):
        super(MktEnvrFwd, self).__init__(target, data_collection, config)

        self.type = 'forward'
