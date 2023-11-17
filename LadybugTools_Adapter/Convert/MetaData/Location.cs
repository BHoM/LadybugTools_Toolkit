using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static BH.oM.LadybugTools.Location ToLocation(Dictionary<string, object> oldObject)
        {
            return new oM.LadybugTools.Location()
            {
                City = (string)oldObject["city"],
                State = (string)oldObject["state"],
                Country = (string)oldObject["country"],
                Latitude = (double)oldObject["latitude"],
                Longitude = (double)oldObject["longitude"],
                TimeZone = (double)oldObject["time_zone"],
                Elevation = (double)oldObject["elevation"],
                StationId = (string)oldObject["station_id"],
                Source = (string)oldObject["source"]
            };
        }

        public static Dictionary<string, object> FromLocation(BH.oM.LadybugTools.Location location)
        {
            return new Dictionary<string, object>()
            {
                { "type", "Location" },
                { "city", location.City },
                { "state", location.State },
                { "country", location.Country },
                { "latitude", location.Latitude },
                { "longitude", location.Longitude },
                { "time_zone", location.TimeZone },
                { "elevation", location.Elevation },
                { "station_id", location.StationId },
                { "source", location.Source }
            };
        }
    }
}
