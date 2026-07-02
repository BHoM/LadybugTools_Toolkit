import json
from typing import Callable
from functools import wraps
from python_toolkit.bhom.bhom_object import BHoMJSONDecoder, BHoMJSONEncoder, BHoMObject

def bhom_callable(argument_types:dict[str, type] = {}, encoder_cls: type = BHoMJSONEncoder, decoder_cls: type = BHoMJSONDecoder):
    """Decorator for functions to be made callable from BHoM C# methods/adapters.

    Note: methods that this wraps must not have "__input_json__" as a kwarg, as this is used internally to allow BHoM adapters to call the method.
    
    Args:
        argument_types (dict[str, type]): this is a dictionary that is used to map the argument names to types (specifically BHoMObject types) to subclasses of BHoMObjects.
            For example, if you have a class that is a subclass of BHoMObject, the default serialiser will only deserialise json to a BHoMObject.
            To go the extra step to get your class, you must provide the type in this dictionary to allow the wrapper to convert the BHoMObject type to your desired type.

        encoder_cls (JSONEncoder): A JSONEncoder (ideally one that is a subclass of BHoMJSONEncoder). Mainly this is for if a custom encoder has been implemented for a specific toolkit.

        decoder_cls (JSONDecoder): same as encoder_cls but for JSONDecoder.
    """
    def decorator(function: Callable):

        #TODO: use module and name to tell run_wrapped what to call?
        print(function.__module__, function.__name__)

        @wraps(function)
        def wrapper(*args, **kwargs):

            do_wrap:bool = False

            if "__input_json__" in kwargs:
                do_wrap = True
                input_json = kwargs.pop("__input_json__")
                #get dictionary from input_file

                if not input_json.startswith("{"): #assume it's a path
                    with open(input_json, "r") as f:
                        input_json = f.read()

                kwargs = json.loads(input_json, cls=decoder_cls)

                for arg_name in argument_types:
                    if arg_name not in kwargs:
                        continue

                    t = argument_types[arg_name]

                    if issubclass(t, BHoMObject) and type(kwargs[arg_name] is BHoMObject):
                        kwargs[arg_name] = t._from_bhom_object(kwargs[arg_name])

            rtn = function(*args, **kwargs)

            if do_wrap:
                json_rtn = json.dumps(rtn, cls=encoder_cls)

                return json_rtn
            
            return rtn
        return wrapper
    return decorator