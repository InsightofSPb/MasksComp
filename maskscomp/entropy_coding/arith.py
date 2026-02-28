from __future__ import annotations

import io
from typing import BinaryIO

import numpy as np


class BitOutputStream:
    def __init__(self, sink: BinaryIO):
        self.sink = sink
        self.current_byte = 0
        self.num_bits_filled = 0
        self.num_bits_written = 0

    def write(self, bit: int) -> None:
        self.current_byte = (self.current_byte << 1) | (bit & 1)
        self.num_bits_filled += 1
        self.num_bits_written += 1
        if self.num_bits_filled == 8:
            self.sink.write(bytes((self.current_byte,)))
            self.current_byte = 0
            self.num_bits_filled = 0

    def flush(self) -> None:
        while self.num_bits_filled != 0:
            self.write(0)


class BitInputStream:
    def __init__(self, source: BinaryIO):
        self.source = source
        self.current_byte = 0
        self.num_bits_remaining = 0

    def read(self) -> int:
        if self.num_bits_remaining == 0:
            b = self.source.read(1)
            if len(b) == 0:
                return 0
            self.current_byte = b[0]
            self.num_bits_remaining = 8
        self.num_bits_remaining -= 1
        return (self.current_byte >> self.num_bits_remaining) & 1


class ArithmeticEncoder:
    STATE_SIZE = 32
    FULL_RANGE = 1 << STATE_SIZE
    HALF_RANGE = FULL_RANGE >> 1
    QUARTER_RANGE = HALF_RANGE >> 1
    MASK = FULL_RANGE - 1

    def __init__(self, bitout: BitOutputStream):
        self.bitout = bitout
        self.low = 0
        self.high = self.MASK
        self.pending = 0

    def write(self, cumul: np.ndarray, symbol: int) -> None:
        total = int(cumul[-1])
        if total <= 0:
            raise ValueError("CDF total must be positive")
        symlow = int(cumul[symbol])
        symhigh = int(cumul[symbol + 1])
        if symlow >= symhigh:
            raise ValueError("Symbol has zero frequency")

        rng = self.high - self.low + 1
        newlow = self.low + (rng * symlow) // total
        newhigh = self.low + (rng * symhigh) // total - 1
        self.low = newlow
        self.high = newhigh

        while ((self.low ^ self.high) & self.HALF_RANGE) == 0:
            bit = self.low >> (self.STATE_SIZE - 1)
            self.bitout.write(int(bit))
            for _ in range(self.pending):
                self.bitout.write(int(bit ^ 1))
            self.pending = 0
            self.low = ((self.low << 1) & self.MASK)
            self.high = ((self.high << 1) & self.MASK) | 1

        while (self.low & ~self.high & self.QUARTER_RANGE) != 0:
            self.pending += 1
            self.low = (self.low << 1) ^ self.HALF_RANGE
            self.high = ((self.high ^ self.HALF_RANGE) << 1) | self.HALF_RANGE | 1

    def finish(self) -> None:
        self.pending += 1
        bit = self.low >> (self.STATE_SIZE - 2)
        self.bitout.write(int(bit))
        for _ in range(self.pending):
            self.bitout.write(int(bit ^ 1))
        self.bitout.flush()


class ArithmeticDecoder:
    STATE_SIZE = ArithmeticEncoder.STATE_SIZE
    FULL_RANGE = ArithmeticEncoder.FULL_RANGE
    HALF_RANGE = ArithmeticEncoder.HALF_RANGE
    QUARTER_RANGE = ArithmeticEncoder.QUARTER_RANGE
    MASK = ArithmeticEncoder.MASK

    def __init__(self, bitin: BitInputStream):
        self.bitin = bitin
        self.low = 0
        self.high = self.MASK
        self.code = 0
        for _ in range(self.STATE_SIZE):
            self.code = (self.code << 1) | self.bitin.read()

    def read(self, cumul: np.ndarray) -> int:
        total = int(cumul[-1])
        if total <= 0:
            raise ValueError("CDF total must be positive")
        rng = self.high - self.low + 1
        value = ((self.code - self.low + 1) * total - 1) // rng
        symbol = int(np.searchsorted(cumul, value, side="right") - 1)
        if symbol < 0 or symbol + 1 >= len(cumul):
            raise ValueError("Decoded symbol out of range")

        symlow = int(cumul[symbol])
        symhigh = int(cumul[symbol + 1])
        newlow = self.low + (rng * symlow) // total
        newhigh = self.low + (rng * symhigh) // total - 1
        self.low = newlow
        self.high = newhigh

        while ((self.low ^ self.high) & self.HALF_RANGE) == 0:
            self.low = ((self.low << 1) & self.MASK)
            self.high = ((self.high << 1) & self.MASK) | 1
            self.code = ((self.code << 1) & self.MASK) | self.bitin.read()

        while (self.low & ~self.high & self.QUARTER_RANGE) != 0:
            self.low = (self.low << 1) ^ self.HALF_RANGE
            self.high = ((self.high ^ self.HALF_RANGE) << 1) | self.HALF_RANGE | 1
            self.code = ((self.code ^ self.HALF_RANGE) << 1) | self.bitin.read()
        return symbol


def encode_to_bytes(write_fn) -> bytes:
    sink = io.BytesIO()
    bitout = BitOutputStream(sink)
    enc = ArithmeticEncoder(bitout)
    write_fn(enc)
    enc.finish()
    return sink.getvalue()
