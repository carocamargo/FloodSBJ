Scripts to reproduce analysis and figures of Camargo et al (under revision), *Do ocean dynamics contribute to coastal floods? A case study of the Shelfbreak Jet and coastal sea level along Southern New England (U.S.)*

To reproduce the data and analysis, you can either run individual scripts (numbered 1-5, and the extra save_xxx.py ones), or you can run run_all.py script. Figures are reproduced in the ```figures.ipynb``` file.
Note:
- To run ```download_TG.py``` and ```save_noaa_thresholds.py``` it's necessary to have internet access. 
-  ```save_waves``` requires that wave buoy data was pre-downloaded from https://www.ndbc.noaa.gov/
-  ```save_era5``` requires that wind stress data was pre-downloaded from https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download
-  ```get_save_data``` will use the tide gauge and era5 data, as well as the SBJ time series, which should already have been downloaded from https://zenodo.org/records/10814048
-  Most scripts will call functions from ```utils_work.py```

No significant changes are expected to happen from the date the repository was published, however small editions to improve the coding might still happen. 
