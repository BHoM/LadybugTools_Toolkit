/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2023, the respective contributors. All rights reserved.
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

using BH.oM.Base.Attributes;
using System.ComponentModel;
using BH.oM.LadybugTools;
using System.Collections.Generic;
using System.Globalization;
using System.Text.RegularExpressions;
using System;

namespace BH.Engine.LadybugTools
{
    public static partial class Convert
    {
        [Description("Convert the case of the given text.")]
        [Input("str", "Text to convert the case of.")]
        [Input("stringCase", "The case to convert the text to.")]
        [Input("ignoreList", "A list of strings to ignore when converting the case.")]
        [Output("str", "Text in \"PascalCase\".")]
        public static string ToCase(this string str, StringCase stringCase = StringCase.Undefined, List<string> ignoreList = null)
        {
            switch (stringCase)
            {
                case StringCase.Pascal:
                    return ToPascalCase(str, ignoreList);
                case StringCase.Snake:
                    return ToSnakeCase(str, ignoreList);
                case StringCase.Camel:
                    return ToCamelCase(str, ignoreList);
                default:
                    BH.Engine.Base.Compute.RecordWarning($"No change to the input ({str}) was made");
                    return str;
            }
        }

        [Description("Convert the case of the keys in the given dictionary.")]
        [Input("dict", "The dictionary whose keys will be converted.")]
        [Input("stringCase", "The case to convert the text to.")]
        [Input("ignoreList", "A list of strings to ignore when converting the case.")]
        [Output("dict", "The dictionary with converted keys.")]
        public static Dictionary<string, object> ToCase(this Dictionary<string, object> dict, StringCase stringCase = StringCase.Undefined, List<string> ignoreList = null)
        {
            switch (stringCase)
            {
                case StringCase.Pascal:
                    return ToPascalCase(dict, ignoreList);
                case StringCase.Snake:
                    return ToSnakeCase(dict, ignoreList);
                case StringCase.Camel:
                    return ToCamelCase(dict, ignoreList);
                default:
                    return dict;
            }
        }   

        [Description("Convert the case of the given text.")]
        [Input("str", "Text to convert the case of.")]
        [Input("ignoreList", "A list of strings to ignore when converting the case.")]
        [Output("str", "Text in \"PascalCase\".")]
        public static string ToPascalCase(this string str, List<string> ignoreList = null)
        {
            if (ignoreList == null)
            {
                ignoreList = new List<string>()
                {
                    "_t",
                    "_bhomVersion",
                };
            }

            if (ignoreList.Contains(str))
            {
                return str;
            }

            TextInfo textInfo = new CultureInfo("en-US", false).TextInfo;
            str = textInfo.ToTitleCase(str);
            return Regex.Replace(str, @"(_)(\w)", m => m.Groups[2].Value.ToUpper());
        }

        [Description("Convert the case of the keys in the given dictionary.")]
        [Input("dict", "The dictionary whose keys will be converted.")]
        [Input("ignoreList", "A list of strings to ignore when converting the case.")]
        [Output("dict", "The dictionary with converted keys.")]
        public static Dictionary<string, object> ToPascalCase(this Dictionary<string, object> dict, List<string> ignoreList = null)
        {
            var convertedDict = new Dictionary<string, object>();
            foreach (var pair in dict)
            {
                var originalKey = ToPascalCase(pair.Key, ignoreList);
                var value = pair.Value;

                if (value is Dictionary<string, object> nestedDict)
                {
                    value = ToPascalCase(nestedDict, ignoreList);
                }
                else if (value is List<object> list)
                {
                    for (int i = 0; i < list.Count; i++)
                    {
                        if (list[i] is Dictionary<string, object> listItemDict)
                        {
                            list[i] = ToPascalCase(listItemDict, ignoreList);
                        }
                    }
                }

                convertedDict.Add(originalKey, value);
            }
            return convertedDict;
        }

        [Description("Convert the case of the given text.")]
        [Input("str", "Text to convert the case of.")]
        [Input("ignoreList", "A list of strings to ignore when converting the case.")]
        [Output("str", "Text in \"snakle_case\".")]
        public static string ToSnakeCase(this string str, List<string> ignoreList = null)
        {
            if (ignoreList == null)
            {
                ignoreList = new List<string>()
                {
                    "_t",
                    "_bhomVersion",
                };
            }

            if (ignoreList.Contains(str))
            {
                return str;
            }

            return Regex.Replace(str, "(?<!^)([A-Z][a-z]|(?<=[a-z])[A-Z])", "_$1").ToLower();
        }

        [Description("Convert the case of the keys in the given dictionary.")]
        [Input("dict", "The dictionary whose keys will be converted.")]
        [Input("ignoreList", "A list of strings to ignore when converting the case.")]
        [Output("dict", "The dictionary with converted keys.")]
        public static Dictionary<string, object> ToSnakeCase(this Dictionary<string, object> dict, List<string> ignoreList = null)
        {
            var convertedDict = new Dictionary<string, object>();
            foreach (var pair in dict)
            {
                var originalKey = ToSnakeCase(pair.Key, ignoreList);
                var value = pair.Value;

                if (value is Dictionary<string, object> nestedDict)
                {
                    value = ToSnakeCase(nestedDict, ignoreList);
                }
                else if (value is List<object> list)
                {
                    for (int i = 0; i < list.Count; i++)
                    {
                        if (list[i] is Dictionary<string, object> listItemDict)
                        {
                            list[i] = ToSnakeCase(listItemDict, ignoreList);
                        }
                    }
                }

                convertedDict.Add(originalKey, value);
            }
            return convertedDict;
        }

        [Description("Convert the case of the given text.")]
        [Input("str", "Text to convert the case of.")]
        [Input("ignoreList", "A list of strings to ignore when converting the case.")]
        [Output("str", "Text in \"camelCase\".")]
        public static string ToCamelCase(this string str, List<string> ignoreList)
        {
            if (ignoreList == null)
            {
                ignoreList = new List<string>()
                {
                    "_t",
                    "_bhomVersion",
                };
            }

            if (ignoreList.Contains(str))
            {
                return str;
            }

            TextInfo textInfo = new CultureInfo("en-US", false).TextInfo;
            str = textInfo.ToTitleCase(str);
            str = Regex.Replace(str, @"(_)(\w)", m => m.Groups[2].Value.ToUpper());

            return Char.ToLowerInvariant(str[0]) + str.Substring(1);
        }

        [Description("Convert the case of the keys in the given dictionary.")]
        [Input("dict", "The dictionary whose keys will be converted.")]
        [Input("ignoreList", "A list of strings to ignore when converting the case.")]
        [Output("dict", "The dictionary with converted keys.")]
        public static Dictionary<string, object> ToCamelCase(this Dictionary<string, object> dict, List<string> ignoreList = null)
        {
            var convertedDict = new Dictionary<string, object>();
            foreach (var pair in dict)
            {
                var originalKey = ToCamelCase(pair.Key, ignoreList);
                var value = pair.Value;

                if (value is Dictionary<string, object> nestedDict)
                {
                    value = ToCamelCase(nestedDict);
                }
                else if (value is List<object> list)
                {
                    for (int i = 0; i < list.Count; i++)
                    {
                        if (list[i] is Dictionary<string, object> listItemDict)
                        {
                            list[i] = ToCamelCase(listItemDict);
                        }
                    }
                }

                convertedDict.Add(originalKey, value);
            }
            return convertedDict;
        }
    }
}

