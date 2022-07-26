# TODO - implement
# raise NotImplementedError()
"""
- Load points and obtain [[x0, y0], [x1, y1], ..., [xn, yn]] array
- Obtain triangular point-point distance matrix (upper/lower triangle only needed as they're identical)
- Obtain triangular point-point vector matrix (upper/lower triangle only needed as they're identical, save for -Ve adjustment)
- Obtain point-point vector angle-to-north, to get the "down-windedness of the point from other points
- Get all unique wind speeds/directions in the year
- For each unique speed/direction, calculate the plume (and magnitude) of points "down-wind" from any source points
- Apply moisture effectivess magnitude to each plume, and run an amax on resultant moisture sets to determine peak effect
- Save moisture matrix as sim_dir / moisture / "moisture_matrix.h5
- Create and save DBT/RH matrices in sim_dir / moisture / ("dry_bulb_temperature.h5"/"relative_humidity.h5")
"""
