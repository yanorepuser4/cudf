# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Datatype utilities."""

from __future__ import annotations

from functools import cache

import pyarrow as pa
from typing_extensions import assert_never

import polars as pl

import cudf._lib.pylibcudf as plc

__all__ = ["from_polars", "to_arrow"]


@cache
def from_polars(dtype: pl.DataType) -> plc.DataType:
    """
    Convert a polars datatype to a pylibcudf one.

    Parameters
    ----------
    dtype
        Polars dtype to convert

    Returns
    -------
    Matching pylibcudf DataType object.

    Raises
    ------
    NotImplementedError for unsupported conversions.
    """
    if isinstance(dtype, pl.Boolean):
        return plc.DataType(plc.TypeId.BOOL8)
    elif isinstance(dtype, pl.Int8):
        return plc.DataType(plc.TypeId.INT8)
    elif isinstance(dtype, pl.Int16):
        return plc.DataType(plc.TypeId.INT16)
    elif isinstance(dtype, pl.Int32):
        return plc.DataType(plc.TypeId.INT32)
    elif isinstance(dtype, pl.Int64):
        return plc.DataType(plc.TypeId.INT64)
    if isinstance(dtype, pl.UInt8):
        return plc.DataType(plc.TypeId.UINT8)
    elif isinstance(dtype, pl.UInt16):
        return plc.DataType(plc.TypeId.UINT16)
    elif isinstance(dtype, pl.UInt32):
        return plc.DataType(plc.TypeId.UINT32)
    elif isinstance(dtype, pl.UInt64):
        return plc.DataType(plc.TypeId.UINT64)
    elif isinstance(dtype, pl.Float32):
        return plc.DataType(plc.TypeId.FLOAT32)
    elif isinstance(dtype, pl.Float64):
        return plc.DataType(plc.TypeId.FLOAT64)
    elif isinstance(dtype, pl.Date):
        return plc.DataType(plc.TypeId.TIMESTAMP_DAYS)
    elif isinstance(dtype, pl.Time):
        raise NotImplementedError("Time of day dtype not implemented")
    elif isinstance(dtype, pl.Datetime):
        if dtype.time_zone is not None:
            raise NotImplementedError("Time zone support")
        if dtype.time_unit == "ms":
            return plc.DataType(plc.TypeId.TIMESTAMP_MILLISECONDS)
        elif dtype.time_unit == "us":
            return plc.DataType(plc.TypeId.TIMESTAMP_MICROSECONDS)
        elif dtype.time_unit == "ns":
            return plc.DataType(plc.TypeId.TIMESTAMP_NANOSECONDS)
        assert dtype.time_unit is not None
        assert_never(dtype.time_unit)
    elif isinstance(dtype, pl.Duration):
        if dtype.time_unit == "ms":
            return plc.DataType(plc.TypeId.DURATION_MILLISECONDS)
        elif dtype.time_unit == "us":
            return plc.DataType(plc.TypeId.DURATION_MICROSECONDS)
        elif dtype.time_unit == "ns":
            return plc.DataType(plc.TypeId.DURATION_NANOSECONDS)
        assert dtype.time_unit is not None
        assert_never(dtype.time_unit)
    elif isinstance(dtype, pl.String):
        return plc.DataType(plc.TypeId.STRING)
    elif isinstance(dtype, pl.Null):
        # TODO: Hopefully
        return plc.DataType(plc.TypeId.EMPTY)
    elif isinstance(dtype, pl.List):
        # TODO: This doesn't consider the value type.
        return plc.DataType(plc.TypeId.LIST)
    else:
        raise NotImplementedError(f"{dtype=} conversion not supported")


@cache
def to_arrow(dtype: plc.DataType) -> pa.DataType:
    """
    Convert a pylibcudf datatype to a pyarrow one.

    Parameters
    ----------
    dtype
        Pylibcudf dtype to convert

    Returns
    -------
    Matching pyarrow type.

    Raises
    ------
    NotImplementedError for unsupported conversions.
    """
    type_id = dtype.id()
    if type_id == plc.types.TypeId.BOOL8:
        return pa.bool_()
    elif type_id == plc.types.TypeId.INT8:
        return pa.int8()
    elif type_id == plc.types.TypeId.INT16:
        return pa.int16()
    elif type_id == plc.types.TypeId.INT32:
        return pa.int32()
    elif type_id == plc.types.TypeId.INT64:
        return pa.int64()
    elif type_id == plc.types.TypeId.UINT8:
        return pa.uint8()
    elif type_id == plc.types.TypeId.UINT16:
        return pa.uint16()
    elif type_id == plc.types.TypeId.UINT32:
        return pa.uint32()
    elif type_id == plc.types.TypeId.UINT64:
        return pa.uint64()
    elif type_id == plc.types.TypeId.FLOAT32:
        return pa.float32()
    elif type_id == plc.types.TypeId.FLOAT64:
        return pa.float64()
    elif type_id == plc.types.TypeId.TIMESTAMP_DAYS:
        return pa.date32()
    elif type_id == plc.types.TypeId.TIMESTAMP_MILLISECONDS:
        return pa.timestamp("ms")
    elif type_id == plc.types.TypeId.TIMESTAMP_MICROSECONDS:
        return pa.timestamp("us")
    elif type_id == plc.types.TypeId.TIMESTAMP_NANOSECONDS:
        return pa.timestamp("ns")
    elif type_id == plc.types.TypeId.DURATION_MILLISECONDS:
        return pa.duration("ms")
    elif type_id == plc.types.TypeId.DURATION_MICROSECONDS:
        return pa.duration("us")
    elif type_id == plc.types.TypeId.DURATION_NANOSECONDS:
        return pa.duration("ns")
    elif type_id == plc.types.TypeId.STRING:
        return pa.string()
    elif type_id == plc.types.TypeId.EMPTY:
        return pa.null()
    else:
        # TODO: LIST needs to provide value_type inside.
        raise NotImplementedError(f"{type_id=} conversion not supported")
