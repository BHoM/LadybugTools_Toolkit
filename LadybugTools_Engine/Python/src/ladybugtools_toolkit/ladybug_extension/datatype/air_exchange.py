# coding=utf-8
"""Air exchange data type."""
from __future__ import division

from ladybug.datatype.base import DataTypeBase


class AirExchange(DataTypeBase):
    """AirExchange"""

    _units = "ach"
    _si_units = "ach"
    _ip_units = "ach"
    _min = float(0)
    _abbreviation = "ach"
    _point_in_time = False
    _cumulative = True

    def to_unit(self, values, unit, from_unit):
        """Return values converted to the unit given the input from_unit."""
        return self._to_unit_base("ach", values, unit, from_unit)

    def to_ip(self, values, from_unit):
        """Return values in IP and the units to which the values have been converted."""
        return values, from_unit

    def to_si(self, values, from_unit):
        """Return values in SI and the units to which the values have been converted."""
        return values, from_unit
