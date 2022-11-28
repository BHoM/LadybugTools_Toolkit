from .analytics import bhom_analytics


@bhom_analytics
def example_function(a: int, b: int) -> int:
    """A decorated example function, returning a result and adding to the
    BHoM Analytics log."""
    return a + b
