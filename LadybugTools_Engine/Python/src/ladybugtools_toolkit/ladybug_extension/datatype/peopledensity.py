# coding=utf-8
"""People density data type."""
from __future__ import division

from ladybug.datatype.base import DataTypeBase


class PeopleDensity(DataTypeBase):
    """PeopleDensity"""

    _units = ("people/m2", "people/ft2")
    _si_units = "people/m2"
    _ip_units = "people/ft2"
    _abbreviation = "PplD"

    def _m2_person_to_ft2_person(self, value):
        return value / 10.7639

    def _ft2_person_to_m2_person(self, value):
        return value * 10.7639

    def to_unit(self, values, unit, from_unit):
        """Return values converted to the unit given the input from_unit."""
        return self._to_unit_base("people/m2", values, unit, from_unit)

    def to_ip(self, values, from_unit):
        """Return values in IP and the units to which the values have been converted."""
        if from_unit in self.ip_units:
            return values, from_unit
        return self.to_unit(values, "people/ft2", from_unit), "people/ft2"

    def to_si(self, values, from_unit):
        """Return values in SI and the units to which the values have been converted."""
        if from_unit in self.si_units:
            return values, from_unit
        return self.to_unit(values, "people/m2", from_unit), "people/m2"
