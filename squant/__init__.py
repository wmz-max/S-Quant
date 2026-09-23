"""S-Quant research reference. No calibration data is used for compression."""
from .config import QuantConfig
from .codec import CompressedTensor, compress, decompress, load, save

__all__ = ["QuantConfig", "CompressedTensor", "compress", "decompress", "save", "load"]
__version__ = "0.1.0"

