# raise NotImplementedError()
# # TODO - Implement
"""_summary_
- Read "cfd" folder from simulation directory
- Load all availabel CFD reuslts (one per angle simulated)
- Construct a lookup from these so that per wind speed and directyion in the EPW, a synthetic hourly CFD results set can be interpolated
- Save wind_speed matrix as sim_dir / cfd / "wind_speed.h5
"""

"""
[
    {
        "pt_velocity_file": "V225.csv",
        "source_velocity": 4.48,
        "wind_direction": 225
    },
    {
        "pt_velocity_file": "V270.csv",
        "source_velocity": 4.94,
        "wind_direction": 270
    },
    {
        "pt_velocity_file": "V315.csv",
        "source_velocity": 4.83,
        "wind_direction": 315
    },
    {
        "pt_velocity_file": "V000.csv",
        "source_velocity": 3.87,
        "wind_direction": 0
    },
    {
        "pt_velocity_file": "V045.csv",
        "source_velocity": 3.565,
        "wind_direction": 45
    },
    {
        "pt_velocity_file": "V090.csv",
        "source_velocity": 3.78,
        "wind_direction": 90
    },
    {
        "pt_velocity_file": "V135.csv",
        "source_velocity": 3.365,
        "wind_direction": 135
    },
    {
        "pt_velocity_file": "V180.csv",
        "source_velocity": 3.16,
        "wind_direction": 180
    }
]
"""
