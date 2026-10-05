"""Manifest-backed NIH report labels with explicit, separate target semantics."""

from src.data.dataset import ChestXRayDataset

NIH_LABEL_TO_ID = {"NO_REPORTED_PNEUMONIA": 0, "REPORT_PNEUMONIA": 1}


class NIHReportDataset(ChestXRayDataset):
    """Use the common image loader without calling other chest diseases NORMAL."""

    label_mapping = NIH_LABEL_TO_ID
