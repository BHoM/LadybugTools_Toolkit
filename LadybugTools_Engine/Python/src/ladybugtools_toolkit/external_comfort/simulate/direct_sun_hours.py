import pandas as pd
from honeybee.model import Model
from ladybug.epw import EPW, AnalysisPeriod
from ladybug.wea import Wea
from lbt_recipes.recipe import Recipe, RecipeSettings


def direct_sun_hours(model: Model, epw: EPW, month: int, day: int, timestep: int = 4) -> pd.DataFrame:


    model = Model.from_hbjson(hbjson)
    analysis_period = AnalysisPeriod(st_month=month, end_month=month, st_day=day, end_day=day, timestep=timestep)
    wea = Wea.from_epw_file(epw.file_path, timestep=timestep).filter_by_analysis_period(analysis_period)
    grid_name = model.properties.radiance.sensor_grids[0].identifier

    recipe = Recipe("direct-sun-hours")
    recipe.input_value_by_name("model", model)
    recipe.input_value_by_name("wea", wea)
    recipe.input_value_by_name("north", 0)
    recipe.input_value_by_name("timestep", timestep)
    recipe.input_value_by_name("grid-filter", grid_name)
    recipe_settings = RecipeSettings(folder=r"C:\Users\tgerrish\Downloads\sdfasdfasdfs")
    proj_dir = recipe.run(
        settings=recipe_settings,
        radiance_check=False,
        queenbee_path=r"C:\Program Files\ladybug_tools\python\Scripts\queenbee.exe",
    )
    return proj_dir#df = load_res_file(pth)


if __name__ == "__main__":

    hbjson = r"C:\Users\tgerrish\simulation\EastHarbour_PROPOSED\EastHarbour_PROPOSED.hbjson"
    model = Model.from_hbjson(hbjson)
    epw = EPW(r"C:\Users\tgerrish\simulation\Toronto_EastHarbour\Coastal_ExternalThermalComfort.epw")

    print(direct_sun_hours(model, epw, 3, 21))
