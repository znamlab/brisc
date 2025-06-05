import flexiznam as flz
import numpy as np
import pandas as pd
from pathlib import Path

from brisc.exploratory_analysis.plot_summary_for_all_bc import (
    project,
    mouse,
    plot_barcode,
)

error_correction_ds_name = "BRAC8498.3e_error_corrected_barcodes_10"
df_file = flz.get_processed_path(
    f"{project}/{mouse}/analysis/{error_correction_ds_name}_cell_barcode_df.pkl"
)

full_df = pd.read_pickle(df_file)
print(f"Loaded {len(full_df)} cells")
save_folder = flz.get_processed_path(f"{project}/{mouse}/analysis/all_starter_cells")
use_slurm = True
slurm_folder = Path.home() / "slurm_logs" / project / mouse / "all_starter_cells"
slurm_folder.mkdir(exist_ok=True, parents=True)

for bc, cells in full_df.groupby("main_barcode"):
    nstart = np.sum(cells.is_starter)
    ncells = len(cells)
    # plot everything that has a starter or 10 cells
    if not np.any(cells.is_starter) and ncells < 10:
        continue
    print(f"Plotting barcode {bc} with {nstart} starters out of {ncells} cells")
    scripts_name = f"plotting_{bc}"
    plot_barcode(
        bc,
        save_folder=save_folder,
        verbose=True,
        plot_raw_data=True,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        scripts_name=scripts_name,
    )
