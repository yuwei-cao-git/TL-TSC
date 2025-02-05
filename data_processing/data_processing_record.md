1. FIR Clean
opt 1: ovf_fri_clean.geometry.area >= 10000 --> 48,417
opt 2: ovf_fri_clean.geometry.area >= 5000 --> 57,100

Select >= 5000

2. remove features overlapped with harvest_ply features
2.1 select by location (processing.run("native:selectbylocation", {'INPUT':'D:/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_fri/ovf_fri_cleaned.shp','PREDICATE':[0,1,5,6,7],'INTERSECT':'D:/Sync/research/tree_species_estimation/tree_dataset/ovf/ovf_fri/Harvest_ply_2014/Harvest_ply.shp','METHOD':0}))
2.2 remove area < 5000

| Steps  | Result |Output filename|
| ------------- | ------------- | ------------- |
| ovf_fri_clean.geometry.area >= 5000  | 57,100  | file:///D:/Sync/research/tree_species_estimation/tree_dataset/ovf/ovf_fri/OVF_2021_PCI/ovf_fri_cleaned.shp  |
| Difference (remove harvest area) | 57,009  | file:///D:/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_fri/ovf_fri_removed_harvest.gpkg  |

2.3 composition selection

> full dataset:
> 
> len(fri_comps_prom) 15800
> 
> Species columns: ['MH', 'IW', 'PO', 'CE', 'BF', 'BD', 'SW', 'BY', 'AB', 'OR', 'MR', 'BE', 'LA', 'BW', 'HE', 'PW', 'AW', 'PR', 'SB', 'OW', 'CB', 'PJ', 'EW', 'HI', 'BN', 'PS', 'OB', 'SR']
> 
> polygons: 57009
> 
```
# Find most prominant species compositions
fri_comps = (
    ovf_fri_clean.groupby(["OSPCOMP"])["POLYID"]  # groupby spcomp
    .count()  # count number of occurances
    .reset_index(name="count")  # reset index
    .sort_values(["count"], ascending=False)  # order decending
)

# Get polygons with more than 20 count
fri_comps = fri_comps[fri_comps["count"] >= 1]
fri_comps_prom = list(fri_comps.OSPCOMP)
```
| Steps  | polygon count |composition count|species count|
| ------------- | ------------- | ------------- | ------------- |
| fri_comps[fri_comps["count"] >= 1  | 57,100  | 15800  | 28  |
| fri_comps[fri_comps["count"] >= 2 | 50810 |  9601  | 28  |
| fri_comps[fri_comps["count"] >= 10 | 19056|  834  | 22 |

2.4. balancing dataset & extract plots
| method  | train  | test |val|
| -------------| ------------- | ------------- | ------------- |
| method1  | 30264  | 3243  | 3243  |
| method2  | 51450  | 5513  | 5513  |

species: ['CE', 'LA', 'BF', 'AB', 'PO', 'BW', 'PW', 'OR', 'SW', 'MR', 'MH', 'IW', 'BE', 'BD', 'PR', 'HE', 'BY', 'SB', 'PJ', 'AW', 'SR', 'PS']

2.5 remove species < 100 among val/test/train & calc perc_specs

| method  | train  | test |val|
| -------------| ------------- | ------------- | ------------- |
| method1  | 27880  | 2993  | 2990  |
| method2  | 48818  | 5230  | 5233  |
| non-balance  | 13628  |  - |  - |

species: ['OR', 'MR', 'BW', 'PO', 'MH', 'SW', 'CE', 'LA', 'BF', 'PW', 'BD', 'BE', 'PR', 'BY', 'SB', 'AB', 'HE', 'IW', 'AW', 'PS', 'PJ', 'SR']

2.5 compare different data balancing & samplling method:
- no data-balancing
![alt text](image-3.png)
- compostion balancing & species count balancing
![alt text](image.png)
- composition balancing & proportions balancing
![alt text](image-1.png)

compare training data:
![alt text](image-2.png)