1.0.4 (in progress)
-------------------
- Fix for floating-point error handling (#55)
- Fix error reading MEME files without "name" per motif (#64)
- Fix get_pair_locations for motif lists with length > 1000 (#62)

1.0.3 (23-01-2023)
------------------
- Added pyproject.toml to fix error introduced in 1.0.2 when installing with pip

1.0.2 (20-01-2023)
-------------------
- Fixed missing readthedocs documentation
- Added a warning if not at least one annotation was found with the given config (annotate_regions())

1.0.1 (24-11-2022)
-------------------
Fixed error when importing mpl_toolkits.axes_grid for matplotlib >= 3.6 (#47); the import now depends on the matplotlib version

1.0.0 (25-10-2022)
-------------------
- Initial release to PyPI
- Start of versioning
