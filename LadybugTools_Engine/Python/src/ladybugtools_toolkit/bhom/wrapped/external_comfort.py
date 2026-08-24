# pylint: disable=C0415,E0401,W0703
from python_toolkit.bhom.decorators import bhom_wrapper
from ladybugtools_toolkit.external_comfort._externalcomfortbase import ExternalComfort
from ladybugtools_toolkit.bhom.from_bhom import LBTBHoMJSONDecoder
from ladybugtools_toolkit.bhom.to_bhom import LBTBHoMJSONEncoder

#Note: All this method does is return the external comfort object given.
#Originally this method converted from json and then back to json, however the bhom_callable decorator does this automatically.
#In order to allow this to still exist as callable from BHoM, this method was simplified to just return.

@bhom_wrapper.bhom_callable("external_comfort", argument_types = {"external_comfort": ExternalComfort}, encoder_cls=LBTBHoMJSONEncoder, decoder_cls=LBTBHoMJSONDecoder)
def main(external_comfort: ExternalComfort, **kwargs) -> None:
    return external_comfort
