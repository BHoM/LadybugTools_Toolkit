# pylint: disable=C0415,E0401,W0703
from python_toolkit.bhom.decorators import bhom_wrapper
from ladybugtools_toolkit.external_comfort._simulatebase import SimulationResult
from ladybugtools_toolkit.bhom.from_bhom import LBTBHoMJSONDecoder
from ladybugtools_toolkit.bhom.to_bhom import LBTBHoMJSONEncoder

#see external_comfort.py

@bhom_wrapper.bhom_callable("simulation_result", argument_types = {"simulation_result": SimulationResult}, encoder_cls=LBTBHoMJSONEncoder, decoder_cls=LBTBHoMJSONDecoder)
def main(simulation_result: SimulationResult, **kwargs) -> None:
    return simulation_result
