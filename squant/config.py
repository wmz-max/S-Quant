from dataclasses import asdict, dataclass

MASKS = {2: 0x3, 3: 0x6, 4: 0xC, 5: 0x14, 6: 0x30, 7: 0x60,
         8: 0xB8, 9: 0x110, 10: 0x240, 11: 0x500, 12: 0xE08,
         13: 0x1C80, 14: 0x3802, 15: 0x6000, 16: 0xB400, 24: 0xE10000}


@dataclass(frozen=True)
class QuantConfig:
    block_size: int = 16
    seed_bits: int = 8
    group_size: int = 32
    energy_threshold: float = 0.9
    min_bases: int = 2
    max_bases: int = 16
    fixed_bases: int | None = None
    seed_budget: int | None = None
    random_seed: int = 0
    seed_chunk: int = 64
    block_chunk: int = 256
    cache_mb: int = 128
    device: str = "numpy"
    strict_threshold: bool = False
    lfsr_mask: int | None = None

    def __post_init__(self):
        for key in ("block_size", "seed_bits", "group_size", "min_bases", "max_bases",
                    "seed_chunk", "block_chunk"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.seed_bits not in MASKS and self.lfsr_mask is None:
            raise ValueError("Unsupported seed width: supply an explicit lfsr_mask")
        if not 2 <= self.seed_bits <= 24:
            raise ValueError("seed_bits must be between 2 and 24")
        if not 0 < self.mask < 2**self.seed_bits:
            raise ValueError("lfsr_mask must fit seed_bits and be nonzero")
        if not 1 <= self.min_bases <= self.max_bases <= self.block_size:
            raise ValueError("Require 1 <= min_bases <= max_bases <= block_size")
        if self.max_bases > 65535:
            raise ValueError("max_bases must fit uint16")
        if self.fixed_bases is not None and not self.min_bases <= self.fixed_bases <= self.max_bases:
            raise ValueError("fixed_bases is outside the basis range")
        if not 0 < self.energy_threshold <= 1:
            raise ValueError("energy_threshold must lie in (0, 1]")
        if self.seed_budget is not None and not 1 <= self.seed_budget < 2**self.seed_bits:
            raise ValueError("seed_budget must be in [1, 2**seed_bits-1]")
        if self.cache_mb < 0:
            raise ValueError("cache_mb cannot be negative")
        if self.device != "numpy" and self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("Search device must be numpy, cpu (PyTorch), or cuda[:index]")

    @property
    def mask(self):
        return self.lfsr_mask if self.lfsr_mask is not None else MASKS[self.seed_bits]

    def to_dict(self):
        return asdict(self)
