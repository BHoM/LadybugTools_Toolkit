from python_toolkit.bhom.bhom_object import BHoMObject, IObject

class PlotInformation(BHoMObject):
    _t: str = "BH.oM.LadybugTools.PlotInformation"
    image: str
    other_data: dict

    def __init__(self, image:str = "", other_data:IObject = None, **kwargs):
        if other_data is None:
            other_data = IObject(_t = "BH.oM.LadybugTools.NoData")

        self.other_data = other_data
        self.image = image

        super().__init__(_t=self._t, **kwargs)