raise NotImplementedError()
# TODO - Implement
"""_summary_
- Read "cfd" folder from simulation directory
- Load all availabel CFD reuslts (one per angle simulated)
- Construct a lookup from these so that per wind speed and directyion in the EPW, a synthetic hourly CFD results set can be interpolated
- Save wind_speed matrix as sim_dir / cfd / "wind_speed.h5
"""
