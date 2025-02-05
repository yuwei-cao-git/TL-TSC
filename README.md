# TL-TSC
## Abstract
## Dataset Generation
### Data
### Ottawa Valley Forest ==2021== Planning Composite Inventory (PCI):
- A FRI is a **polygon** level snapshot of a Forest Management Unit's land and water features, including **a detailed description of individual forest stands**. Stand attributes include things such as *forest type, tree species, height and age*.  designed as a large-scale survey that would allow general characterization of the forest in terms of species, forest conditions and regeneration
- A [PCI](!https://library.mcmaster.ca/maps/geospatial/forest-resources-planning-composite-inventory-fri-pci#:~:text=A%20planning%20composite%20inventory%20includes,for%20forest%20management%20planning%20purposes) includes **other spatial layers such as ownership and administrative boundaries** to assist in analysis/summarization for forest management *planning* purposes. 
  - year: 2021
  - polygons: 146,991
  - composition: 
    - OSPCOMP (overstorey species composition) - 85063 Polygons that contains OSPCOMP
    - USPCOMP (understorey species composition) 
- Harvest_ply_2014
  - FRI work done by JWRL, file name Harvest_ply - polyline

### SPL data:
- ROI extraction from forest management unit polygon (FORMGMT-LIO-2023008-19) - ROI_OVF
- clip FRI_Leaf_On_Tile_index_shp using the roi of OVF -> output [csv file](./data_processing/FRI_Tile_Index_OVF.csv)
- [script](./data_processing/download_SPL.py) for automatic downloading the data
  - files: 1kmZ177500505202023L.laz -> 1km grid size, 17750050520 -> id, 2023L-> acquire time
  - time range: 2018-2023, ==growing season, leaf-on conditions, Jun. - Sep.==
    > 2018: 221
    > 2019: 559
    > 2020: 28
    > 2022: 8460
    > 2023: 113
  - density: 25 points/m2

### Pseudo-plots generating
- polygons filters
  - land classification types: productive polygons only -> `"POLYTYPE" = 'FOR'` -> result in 85,063 polygons
      > Areas that are capable of producing trees and can support tree growth. These areas may or may not be capable of supporting the harvesting of timber on a sustained yield basis. Some areas may have physical and/or biological characteristics which effect land use. Thus this polygon type includes both production and protection forest areas
  - minimum area of 10,000 m2 -> ` "POLYTYPE" = 'FOR' AND  "Shape_Area" >= 10000`
      > To mitigate edge effects between small adjacent forest stands.
  - a minimum of `20` unique polygons sharing the same composition -> promint species composition
    - calculate species composition count
    - a minimum of 20 unique polygons sharing the same composition

- generate pseduo plots
  - generating number of percentage of each specie (only ovf needed)
  - randomly sampling plots in each cleaned polygon & filter plots within 50m from the boundary of each polygon
  - remove non-common species
    - remove species show less than `100` in plots
    - generate `perc_specs` field to represent the percentage of common species

- Filter Natural Forest Disturbances
  - remove harvest areas (overlapped with "harvest_ply")
  - NTEMS mask