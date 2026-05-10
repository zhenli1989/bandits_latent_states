from src.utils import LIST_FEATURE_DUMMY

class AppearEstimator(object):
    __LIST_FEATURES__ = LIST_FEATURE_DUMMY
    def __init__(self, dict_coef):
        self.dict_coef = dict_coef

    def predict(self, df):
        output = 0
        for var_ in self.__LIST_FEATURES__:
            output += df[var_] * self.dict_coef[var_]

        return output
