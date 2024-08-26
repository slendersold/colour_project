from functools import partial
import os.path
from typing import Literal

import numpy as np
import pandas as pd

from colour import (
    MSDS_CMFS,
    SDS_ILLUMINANTS,
    MultiSpectralDistributions,
    SpectralShape,
    gamma_function,
    msds_to_XYZ,
)
from colour.models.rgb import RGB_COLOURSPACES, XYZ_to_RGB


class RawDataParser:
    ILLUMINANT = Literal["D50", "D65"]
    COLSPACE = Literal["sRGB", "NTSC (1987)"]

    def __init__(
        self,
        reference_basepath="./calibration_data/",
        wl_min=340,
        wl_max=830,
        wl_step=5,
    ):
        self.reference_basepath = reference_basepath
        self.ref_patch_order, self.msds = self.load_msds(
            f"{reference_basepath}/wavelengths_{wl_step}nmstep.csv",
            wl_min,
            wl_max,
            wl_step,
        )
        self.charts = {
            "D50": self.calculate_xyz_chart(
                observer=MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
                illuminant=SDS_ILLUMINANTS["D50"],
            ),
            "D65": self.calculate_xyz_chart(
                observer=MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
                illuminant=SDS_ILLUMINANTS["D65"],
            ),
        }
        self.charts["sRGB"] = self.calculate_rgb_chart(self.charts["D65"], "sRGB")
        self.charts["NTSC (1987)"] = self.calculate_rgb_chart(
            self.charts["D65"], "NTSC (1987)"
        )

    def load_msds(self, config_path, wl_min, wl_max, wl_step):
        wvl_df = pd.read_csv(config_path)
        wvl_df["Wavelength"] = wvl_df["Wavelength"].astype(np.uint16)
        assert (
            wvl_df.Wavelength.iloc[0] == wl_min
        ), "passed min value of wavelength does not match csv stored at passed base path"
        assert (
            wvl_df.Wavelength.iloc[-1] == wl_max
        ), "passed max value of wavelength does not match csv stored at passed base path"
        assert (
            wvl_df.Wavelength.diff().mode().item() == wl_step
        ), "passed step value of wavelengths does not match csv stored at passed base path"

        chart_sd = wvl_df.set_index("Wavelength")

        return (
            chart_sd.columns.to_list(),
            MultiSpectralDistributions(
                chart_sd.to_numpy(), SpectralShape(wl_min, wl_max, wl_step)
            ),
        )

    def calculate_xyz_chart(
        self,
        illuminant=SDS_ILLUMINANTS["D65"],
        observer=MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
    ):
        xyzs = (
            msds_to_XYZ(
                self.msds, cmfs=observer, illuminant=illuminant, method="Integration"
            )
            / 100
        )
        assert xyzs.max() < 1
        return xyzs

    def calculate_rgb_chart(self, xyz_chart, colorspace: COLSPACE = "sRGB"):
        assert colorspace in ["sRGB", "NTSC (1987)", "DON RGB 4"]

        _colspace = RGB_COLOURSPACES[colorspace]
        if colorspace == "NTSC (1987)":  # 'NTSC (1987)', 1953
            _colspace._cctf_decoding = partial(gamma_function, exponent=1.8)
            _colspace._cctf_encoding = partial(gamma_function, exponent=1 / 1.8)

        ret = np.zeros_like(xyz_chart)
        for idx, p in enumerate(xyz_chart):
            ret[idx] = XYZ_to_RGB(
                XYZ=p, colourspace=_colspace, apply_cctf_encoding=False
            )
        ret = np.nan_to_num(ret)
        ret[ret < 0] = 0
        return ret

    def get_reference_xyz(self, illuminant: ILLUMINANT = "D65"):
        return self.charts[illuminant]

    def get_reference_rgb(self, colour_space: COLSPACE = "sRGB"):
        return self.charts[colour_space]

    def save_xyz_values(self):
        np.savez(
            f"{self.reference_basepath}/xyz_values.npz",
            D50=self.get_reference_d50(),
            D65=self.get_reference_d65(),
        )
        print(
            f"generated D65 and D50 XYZ values for 2-degree observer, saved to {self.reference_basepath}/xyz_values.npz"
        )

    def save_rgb_values(self):
        np.savez(
            f"{self.reference_basepath}/rgb_values.npz",
            sRGB=self.get_reference_srgbs(),
            DON4=self.get_reference_don4s(),
            NTSC=self.get_reference_ntscs(),
        )
        print(
            f"generated sRGB, DON RGB 4 and NTSC values, saved to {self.reference_basepath}/rgb_values.npz"
        )
