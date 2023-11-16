using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static BH.oM.LadybugTools.Location Location(BH.Adapter.LadybugTools.Location oldObject)
        {
            return new oM.LadybugTools.Location()
            {
                City = oldObject.City,
                State = oldObject.State,
                Country = oldObject.Country,
                Latitude = oldObject.Latitude,
                Longitude = oldObject.Longitude,
                TimeZone = oldObject.TimeZone,
                Elevation = oldObject.Elevation,
                StationId = oldObject.StationId,
                Source = oldObject.Source
            };
        }
    }
}
