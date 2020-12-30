# ===============================================================
# =========== some enumerates used in this program ===============
# ===============================================================
from enum import Enum, auto


class MutualAttender(Enum):
    NONE = auto()
    ML_MMA = auto()  # (MultiLayer Mutual Multi-Head Attention)


class IntegrationAttender(Enum):
    NONE = auto()
    ML_MA = auto()  # (MultiLayer Multi-Head Attention)


class PoolingStyle(Enum):
    mean = auto()
    max = auto()
    mean_max = auto()
    attn = auto()


class REDataset(Enum):
    DocRED = auto()
    CDR = auto()
