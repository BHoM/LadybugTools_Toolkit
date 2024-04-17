/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2024, the respective contributors. All rights reserved.
 *
 * Each contributor holds copyright over their respective contributions.
 * The project versioning (Git) records all such contribution source information.
 *                                           
 *                                                                              
 * The BHoM is free software: you can redistribute it and/or modify         
 * it under the terms of the GNU Lesser General Public License as published by  
 * the Free Software Foundation, either version 3.0 of the License, or          
 * (at your option) any later version.                                          
 *                                                                              
 * The BHoM is distributed in the hope that it will be useful,              
 * but WITHOUT ANY WARRANTY; without even the implied warranty of               
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the                 
 * GNU Lesser General Public License for more details.                          
 *                                                                            
 * You should have received a copy of the GNU Lesser General Public License     
 * along with this code. If not, see <https://www.gnu.org/licenses/lgpl-3.0.html>.      
 */

using BH.Engine.Base;
using BH.Engine.Environment;
using BH.Engine.Geometry;
using BH.oM.Base.Attributes;
using BH.oM.Environment.Elements;
using BH.oM.Geometry;
using BH.oM.LadybugTools;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Security.Cryptography;

namespace BH.Engine.LadybugTools
{
    public static partial class Create
    {
        [Description("Create a Typology object.")]
        [Input("identifier", "The identifier of the typology.")]
        [Input("shelters", "The shelters of the typology.")]
        [Input("evaporativeCoolingEffect", "A list of hourly-annual dimensionless values by which to adjust the additional of moisture into the air and modify the dry-bulb temperature and relative humidity values. A value 0 means no additional moisure added to air, wheras a value of 1 results in fully moisture saturated air at 100% relative humidity.")]
        [Input("targetWindSpeed", "The hourly target wind speed of the typology, in m/s. This can also contain \"null\" values in which case the EPW file used alongside this object and the porosity of the shelters will be used to determine wind speed - otherwise, any value input here will overwrite those calculated wind speeds.")]
        [Input("radiantTemperatureAdjustment", "A list of values in �C, one-per-hour to adjust the mean radiant temperature by.")]
        [Output("typology", "A Typology object.")]
        public static Typology Typology(
            string identifier = null,
            List<Shelter> shelters = null,
            List<double> evaporativeCoolingEffect = null,
            List<double?> targetWindSpeed = null,
            List<double> radiantTemperatureAdjustment = null
        )
        {
            shelters = shelters ?? new List<Shelter>();

            if ((evaporativeCoolingEffect.Count() == 0 && evaporativeCoolingEffect.Sum() == 0) || evaporativeCoolingEffect == null)
                evaporativeCoolingEffect = Enumerable.Repeat(0.0, 8760).ToList();

            if (evaporativeCoolingEffect.Count() != 8760)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(evaporativeCoolingEffect)} must be a list of 8760 values.");
                return null;
            }

            if (evaporativeCoolingEffect.Where(x => x < 0 || x > 1).Any())
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(evaporativeCoolingEffect)} must be between 0 and 1.");
                return null;
            }

            if ((targetWindSpeed.Count() == 0 && targetWindSpeed.Sum() == 0) || targetWindSpeed == null)
                targetWindSpeed = Enumerable.Repeat<double?>(null, 8760).ToList();

            if (targetWindSpeed.Count() != 8760)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(targetWindSpeed)} must be a list of 8760 values.");
                return null;
            }

            if (targetWindSpeed.Where(x => x != null && x.Value < 0).Any())
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(targetWindSpeed)} values must be greater than or equal to 0, or null if not relevant for that hour of the year.");
                return null;
            }

            if ((radiantTemperatureAdjustment.Count() == 0 && radiantTemperatureAdjustment.Sum() == 0) || radiantTemperatureAdjustment == null)
                radiantTemperatureAdjustment = Enumerable.Repeat(0.0, 8760).ToList();

            if (radiantTemperatureAdjustment.Count() != 8760)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(radiantTemperatureAdjustment)} must be a list of 8760 values.");
                return null;
            }

            if (identifier == null)
            {
                dynamic targetWindSpeedAvg;
                if (targetWindSpeed.Where(x => x.HasValue).Count() == 0)
                {
                    targetWindSpeedAvg = "EPW";
                }
                else
                {
                    targetWindSpeedAvg = targetWindSpeed.Where(x => x.HasValue).Average(x => x.Value);
                }
                if (shelters.Count() == 0)
                {
                    identifier = $"ec{evaporativeCoolingEffect.Average()}_ws{targetWindSpeedAvg}_mrt{radiantTemperatureAdjustment.Average()}";
                }
                else
                {
                    identifier = $"shelters{shelters.Count()}_ec{evaporativeCoolingEffect.Average()}_ws{targetWindSpeedAvg}_mrt{radiantTemperatureAdjustment.Average()}";
                }
                Base.Compute.RecordNote($"This typology has been automatically named \"{identifier}\".");
            }

            return new Typology()
            {
                Name = identifier,
                Shelters = shelters,
                EvaporativeCoolingEffect = evaporativeCoolingEffect,
                TargetWindSpeed = targetWindSpeed,
                RadiantTemperatureAdjustment = radiantTemperatureAdjustment
            };
        }

        /******************************************************/

        [Description("Create a typology from a set of inputs.")]
        [Input("name", "The identifier of the typology created.")]
        [Input("hasSkyShelter", "Whether the typology has a circular sky shelter.")]
        [Input("skyShelterRadius", "The Radius of the circular sky shelter.")]
        [Input("skyShelterHeight", "The height the sky shelter is above the ground.")]
        [Input("hasNorthShelter", "Whether the typology has a north-facing shelter wall.")]
        [Input("northShelterHeight", "The height that the northern shelter wall extends to.")]
        [Input("hasEastShelter", "Whether the typology has an east-facing shelter wall.")]
        [Input("eastShelterHeight", "The height that the eastern shelter wall extends to.")]
        [Input("hasSouthShelter", "Whether the typology has a south-facing shelter wall.")]
        [Input("southShelterHeight", "The eight that the southern shelter wall extends to.")]
        [Input("hasWestShelter", "Whether the typology has a west-facing shelter wall.")]
        [Input("westShelterHeight", "The height that the western shelter wall extends to.")]
        [Input("shelterWindPorosity", "The wind porosity of all shelters (must be between 0 and 1 inclusive).")]
        [Input("shelterRadiationPorosity", "The radiation porosity of all shelters (must be between 0 and 1 inclusive).")]
        [Input("evaporativeCoolingEffect", "The effective evaporative cooling that is present in this typology (must be between 0 and 1 inclusive).")]
        [Input("radiantTemperatureAdjustment", "Any radiant temperature adjustment to apply to the resultant typology.")]
        [Input("targetWindSpeed", "List of wind speeds to be used in this typology (will replace wind speeds present in the epw when calculating UTCI).")]
        [Output("typology", "The generated typology.")]
        public static Typology Typology(
            string name,
            bool hasSkyShelter = false,
            double skyShelterRadius = 5,
            double skyShelterHeight = 2.5,
            bool hasNorthShelter = false,
            double northShelterHeight = 2.5,
            bool hasEastShelter = false,
            double eastShelterHeight = 2.5,
            bool hasSouthShelter = false,
            double southShelterHeight = 2.5,
            bool hasWestShelter = false,
            double westShelterHeight = 2.5,
            double shelterWindPorosity = 0,
            double shelterRadiationPorosity = 0,
            double evaporativeCoolingEffect = 0,
            double radiantTemperatureAdjustment = 0,
            double? targetWindSpeed = null
        )
        {
            if (hasSkyShelter && skyShelterRadius < 0)
            {
                BH.Engine.Base.Compute.RecordError("The sky shelter radius must be greater than or equal to 0.");
                return null;
            }

            if (hasSkyShelter && skyShelterHeight < 0)
            {
                BH.Engine.Base.Compute.RecordError("The sky shelter height must be greater than or equal to 0.");
                return null;
            }
            if (hasNorthShelter && northShelterHeight < 0)
            {
                BH.Engine.Base.Compute.RecordError("The north shelter height must be greater than or equal to 0.");
                return null;
            }
            if (hasEastShelter && eastShelterHeight < 0)
            {
                BH.Engine.Base.Compute.RecordError("The east shelter height must be greater than or equal to 0.");
                return null;
            }
            if (hasSouthShelter && southShelterHeight < 0)
            {
                BH.Engine.Base.Compute.RecordError("The south shelter height must be greater than or equal to 0.");
                return null;
            }
            if (hasWestShelter && westShelterHeight < 0)
            {
                BH.Engine.Base.Compute.RecordError("The west shelter height must be more than or equal to 0.");
                return null;
            }

            if (shelterWindPorosity < 0 || shelterWindPorosity > 1)
            {
                BH.Engine.Base.Compute.RecordError("The shelter wind porosity must be between 0 and 1 inclusive.");
                return null;
            }
            if (shelterRadiationPorosity < 0 || shelterRadiationPorosity > 1)
            {
                BH.Engine.Base.Compute.RecordError("The shelter radiation porosity must be between 0 and 1 inclusive.");
                return null;
            }
            if (evaporativeCoolingEffect < 0 || evaporativeCoolingEffect > 1)
            {
                BH.Engine.Base.Compute.RecordError("The evaporative cooling effect must be between 0 and 1 inclusive.");
                return null;
            }
            if (targetWindSpeed != null && targetWindSpeed < 0)
            {
                BH.Engine.Base.Compute.RecordError("The target wind speed must be more than or equal to 0, or be null.");
                return null;
            }

            Polyline perimeter = Geometry.Create.Polyline(new List<Point>() { Geometry.Create.Point(x: 1, y: 1), Geometry.Create.Point(x: 1, y: -1), Geometry.Create.Point(x: -1, y: -1), Geometry.Create.Point(x: -1, y: 1), Geometry.Create.Point(x: 1, y: 1) });
            List<Panel> panels = Environment.Compute.ExtrudeToVolume(perimeter, "temp", 2).Where(x => x.Type == PanelType.Wall).ToList();

            List<double> radiationPorosity = Enumerable.Repeat(shelterRadiationPorosity, 8760).ToList();
            List<double> windPorosity = Enumerable.Repeat(shelterWindPorosity, 8760).ToList();
            List<double> radiantTemperature = Enumerable.Repeat(radiantTemperatureAdjustment, 8760).ToList();
            List<double> evaporativeCooling = Enumerable.Repeat(evaporativeCoolingEffect, 8760).ToList();
            List<double?> windSpeed = Enumerable.Repeat(targetWindSpeed, 8760).ToList();
            List<Shelter> shelters = new List<Shelter>();

            if (hasEastShelter)
                shelters.Add(Convert.ToShelter(panels[0].ChangeHeight(eastShelterHeight), radiationPorosity, windPorosity));
            if (hasSouthShelter)
                shelters.Add(Convert.ToShelter(panels[1].ChangeHeight(southShelterHeight), radiationPorosity, windPorosity));
            if (hasWestShelter)
                shelters.Add(Convert.ToShelter(panels[2].ChangeHeight(westShelterHeight), radiationPorosity, windPorosity));
            if (hasNorthShelter)
                shelters.Add(Convert.ToShelter(panels[3].ChangeHeight(northShelterHeight), radiationPorosity, windPorosity));

            if (hasSkyShelter)
            {
                shelters.Add(
                    Convert.ToShelter(
                        new Panel() { ExternalEdges = Geometry.Create.Circle(Geometry.Create.Point(x: 0, y: 0, z: skyShelterHeight), skyShelterRadius).CollapseToPolyline(Tolerance.Angle, 36).ToEdges() },
                        radiationPorosity,
                        windPorosity
                        )
                    );
            }

            return Create.Typology(name, shelters, evaporativeCooling, windSpeed, radiantTemperature);
        }

        private static Panel ChangeHeight(this Panel panel, double height)
        {
            Polyline pLine = panel.Polyline();
            //Change Z values of control points that have a z value that isn't equal to 0
            pLine.ControlPoints = pLine.ControlPoints.Select(x => Geometry.Create.Point(x: x.X, y: x.Y, z: x.Z == 0 ? 0 : height)).ToList();
            return new Panel() { ExternalEdges = pLine.ToEdges() };
        }
    }
}