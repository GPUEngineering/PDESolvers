from enum import Enum

class OptionType(Enum):
    EUROPEAN_CALL = 'European Call'
    EUROPEAN_PUT = 'European Put'

class Greeks(Enum):
    DELTA = 'Delta'
    GAMMA = 'Gamma'
    THETA = 'Theta'