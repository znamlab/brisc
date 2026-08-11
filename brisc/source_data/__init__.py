"""Source Data packaging for the BRISC manuscript.

Builds Nature Communications-compliant Excel workbooks (one worksheet per figure
panel) and can redraw each panel back from those workbooks.
"""

from brisc.source_data.figures import (
    build_fig1_source_data,
    build_fig2_source_data,
    build_fig3_source_data,
    build_fig4_source_data,
    build_fig5_source_data,
    build_fig6_source_data,
    export_fig1_source_data,
    export_fig2_source_data,
    export_fig3_source_data,
    export_fig4_source_data,
    export_fig5_source_data,
    export_fig6_source_data,
)
from brisc.source_data.io import (
    extract_counts_array,
    read_source_data_workbook,
    save_excel_sheets,
)
from brisc.source_data.supplementary import (
    build_suppfig4_source_data,
    build_suppfig6_source_data,
    build_suppfig_reviewer_source_data,
    export_suppfig2_source_data,
    export_suppfig4_source_data,
    export_suppfig5_source_data,
    export_suppfig6_source_data,
    export_suppfig8_source_data,
    export_suppfig9_source_data,
    export_suppfig_reviewer_source_data,
)

__all__ = [
    "build_fig1_source_data",
    "build_fig2_source_data",
    "build_fig3_source_data",
    "build_fig4_source_data",
    "build_fig5_source_data",
    "build_fig6_source_data",
    "build_suppfig4_source_data",
    "build_suppfig6_source_data",
    "build_suppfig_reviewer_source_data",
    "export_fig1_source_data",
    "export_fig2_source_data",
    "export_fig3_source_data",
    "export_fig4_source_data",
    "export_fig5_source_data",
    "export_fig6_source_data",
    "export_suppfig2_source_data",
    "export_suppfig4_source_data",
    "export_suppfig5_source_data",
    "export_suppfig6_source_data",
    "export_suppfig8_source_data",
    "export_suppfig9_source_data",
    "export_suppfig_reviewer_source_data",
    "extract_counts_array",
    "read_source_data_workbook",
    "save_excel_sheets",
    "plot_all_workbooks",
    "plot_sheet",
    "plot_workbook",
]


def __getattr__(name):
    # Plotting pulls in matplotlib; keep it out of the import path of the exporters.
    if name in ("plot_all_workbooks", "plot_sheet", "plot_workbook"):
        from brisc.source_data import panels

        return getattr(panels, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
