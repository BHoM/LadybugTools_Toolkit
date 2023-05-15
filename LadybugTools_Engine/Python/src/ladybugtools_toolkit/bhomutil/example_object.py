# pylint: disable=no-member

from .bhom_object import BHoMObject
from .example_function import example_function


class ExampleObject(BHoMObject):
    """An example object inheriting from BHoMObject."""

    # pylint disable=no-member
    def __init__(self, a: int):
        self.a = a
        super().__init__()

    def example_object_function(self) -> int:
        """A dummy method used to demonstrate automatic application of BHoM
        analytics via decoration."""
        return 12

    def another_example_object_function(self) -> int:
        """A method demonstrating the calling of another method, accessing the
        original form, not the wrapped form"""
        return self.example_object_function.__wrapped__() + 3

    def imported_example_function(self) -> int:
        """An imported function that was decorated where it was defined. It is
        wrapped here to avoid double counting it's usage."""
        return example_function.__wrapped__(1, 2)

    @property
    def example_property(self) -> str:
        """An example property."""
        return "an example property!"

    @classmethod
    def from_int(cls, num: int):
        """An example class method."""
        return cls(num)

    # pylint enable=no-member
