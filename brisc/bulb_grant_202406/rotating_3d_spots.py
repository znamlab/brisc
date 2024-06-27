import pandas as pd
import matplotlib as mpl
from itertools import cycle
import napari

# it works better locally. Download csvs and put path here
rabies_spots_csv = "/Users/blota/Data/processed/becalia_rabies_barseq/BRAC8498.3e/analysis/registered_rabies/rabies_spots.csv"
rabies_cell_csv = "/Users/blota/Data/processed/becalia_rabies_barseq/BRAC8498.3e/analysis/registered_rabies/rabies_cells.csv"


rab_df = pd.read_csv(rabies_spots_csv)
ok_sp = rab_df[rab_df["cell_mask"] != -1]

rab_cells = pd.read_csv(rabies_cell_csv)
starter = rab_cells[(rab_cells["starter"]) & (rab_cells["max_n_spots"] > 10)]
starter.main_barcode.value_counts() != 1
viewer = napari.Viewer()
print("Adding bg")
viewer.add_points(
    ok_sp[["z_reg_um", "y_reg_um", "x_reg_um"]].values,
    name="Assigned rolonies",
    size=0.5,
    opacity=0.1,
    face_color="w",
    edge_color="w",
)

colors = cycle(mpl.colormaps["Set1"].colors)
for ist, start in enumerate(starter.index[::20]):
    if ist > 10:
        break
    print(start)
    prop = rab_cells.loc[start]
    col = next(colors)
    viewer.add_points(
        prop[["z_reg_um", "y_reg_um", "x_reg_um"]].values,
        name=prop["main_barcode"],
        size=50,
        face_color=[col],
        edge_color="w",
    )
    rabies_spots_csv = rab_df[rab_df["corrected_bases"] == prop["main_barcode"]]
    sp_ass = rabies_spots_csv[rabies_spots_csv["cell_mask"] != -1]
    viewer.add_points(
        sp_ass[["z_reg_um", "y_reg_um", "x_reg_um"]].values,
        name=f"{prop['main_barcode']} assigned spots",
        size=10,
        face_color=[col],
        edge_color="none",
        opacity=0.8,
    )
    sp_noass = rabies_spots_csv[rabies_spots_csv["cell_mask"] == -1]
    viewer.add_points(
        sp_noass[["z_reg_um", "y_reg_um", "x_reg_um"]].values,
        name=f"{prop['main_barcode']} not assigned spots",
        size=1,
        face_color=[col],
        edge_color="none",
        opacity=0.5,
    )


napari.run()
print("do")
