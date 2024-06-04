from functools import partial

import numpy as np
import pandas as pd

from colour import (MSDS_CMFS, SDS_ILLUMINANTS, MultiSpectralDistributions,
                    SpectralShape, gamma_function, msds_to_XYZ)
from colour.models.rgb import RGB_COLOURSPACES, XYZ_to_RGB

default_colref_basepath = '.'

class RawDataParser:
    def __init__(self, reference_basepath='./calibration_data/', wl_min=340, wl_max=830, wl_step=5):
        self.reference_basepath = reference_basepath
        self.wl_min, self.wl_max, self.wl_step = wl_min, wl_max, wl_step
        self.ref_patch_order, self.msds = self.load_msds(f'{reference_basepath}/wavelengths_{wl_step}nmstep.csv',
                                                         wl_min, wl_max, wl_step)

    def load_msds(self, config_path, wl_min, wl_max, wl_step):
        wvl_df = pd.read_csv(config_path)
        wvl_df['Wavelength'] = (wvl_df['Wavelength']
                                .astype(np.uint16)
                                )
        assert wvl_df.Wavelength.iloc[
                   0] == wl_min, "passed min value of wavelength does not match csv stored at passed base path"
        assert wvl_df.Wavelength.iloc[
                   -1] == wl_max, "passed max value of wavelength does not match csv stored at passed base path"
        assert wvl_df.Wavelength.diff().mode().item() == wl_step, "passed step value of wavelengths does not match csv stored at passed base path"

        chart_sd = wvl_df.set_index('Wavelength')

        return (chart_sd.columns.to_list(),
                MultiSpectralDistributions(chart_sd.to_numpy(),
                                           SpectralShape(wl_min, wl_max, wl_step))
                )

    def calculate_xyz_chart(self, msds,
                            observer=MSDS_CMFS['CIE 1931 2 Degree Standard Observer'],
                            illuminant=SDS_ILLUMINANTS['D65']):
        xyzs = msds_to_XYZ(msds,
                           cmfs=observer,
                           illuminant=illuminant,
                           method='Integration'
                           ) / 100
        assert xyzs.max() < 1
        return xyzs

    def calculate_rgb_chart(self, xyz_chart, colorspace='sRGB'):
        assert colorspace in ['sRGB', 'NTSC (1987)', 'DON RGB 4']

        _colspace = RGB_COLOURSPACES[colorspace]
        if colorspace == 'NTSC (1987)':  # 'NTSC (1987)', 1953
            _colspace._cctf_decoding = partial(gamma_function, exponent=1.8)
            _colspace._cctf_encoding = partial(gamma_function, exponent=1 / 1.8)

        ret = np.zeros_like(xyz_chart)
        for idx, p in enumerate(xyz_chart):
            ret[idx] = XYZ_to_RGB(
                XYZ=p,
                colourspace=_colspace,
                apply_cctf_encoding=True
            )
        ret = np.nan_to_num(ret)
        ret[ret < 0] = 0
        return ret

    def get_reference_d50(self):
        return self.calculate_xyz_chart(self.msds,
                                        observer=MSDS_CMFS['CIE 1931 2 Degree Standard Observer'],
                                        illuminant=SDS_ILLUMINANTS['D50'])

    def get_reference_d65(self):
        return self.calculate_xyz_chart(self.msds,
                                        observer=MSDS_CMFS['CIE 1931 2 Degree Standard Observer'],
                                        illuminant=SDS_ILLUMINANTS['D65'])

    def get_reference_srgbs(self):
        return self.calculate_rgb_chart(self.get_reference_d65(), 'sRGB')

    def get_reference_don4s(self):
        return self.calculate_rgb_chart(self.get_reference_d50(), 'DON RGB 4')

    def get_reference_ntscs(self):
        return self.calculate_rgb_chart(self.get_reference_d65(), 'NTSC (1987)')

    def save_xyz_values(self):
        np.savez(f"{self.reference_basepath}/xyz_values.npz",
                 D50=self.get_reference_d50(),
                 D65=self.get_reference_d65())
        print(f"generated D65 and D50 XYZ values for 2-degree observer, saved to {self.reference_basepath}/xyz_values.npz")

    def save_rgb_values(self):
        np.savez(f"{self.reference_basepath}/rgb_values.npz",
                 sRGB=self.get_reference_srgbs(),
                 DON4=self.get_reference_don4s(),
                 NTSC=self.get_reference_ntscs())
        print(f"generated sRGB, DON RGB 4 and NTSC values, saved to {self.reference_basepath}/rgb_values.npz")