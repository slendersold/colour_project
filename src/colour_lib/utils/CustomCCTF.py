from colour.utilities import CanonicalMapping
from colour.models.rgb import transfer_functions
from colour.models.rgb.transfer_functions import gamma_function
from functools import partial
from typing import Literal


class CustomCCTF:
    MODE = Literal["encode", "decode"]

    def __init__(self):
        self.TF = transfer_functions
        CUSTOM_DECODINGS: CanonicalMapping = CanonicalMapping(
            {
                "Gamma 1.0": partial(gamma_function, exponent=[1.0, 1.0, 1.0]),
                "Gamma 1.8": partial(gamma_function, exponent=[1.8, 1.8, 1.8]),
                "Gamma 1.1, 1.2, 1.9": partial(
                    gamma_function, exponent=[1.1, 1.2, 1.9]
                ),
            }
        )
        self.TF.CCTF_DECODINGS.update(CUSTOM_DECODINGS)
        CUSTOM_ENCODINGS: CanonicalMapping = CanonicalMapping(
            {
                "Gamma 1.0": partial(
                    gamma_function, exponent=[1.0 / 1.0, 1.0 / 1.0, 1.0 / 1.0]
                ),
                "Gamma 1.8": partial(
                    gamma_function, exponent=[1.0 / 1.8, 1.0 / 1.8, 1.0 / 1.8]
                ),
                "Gamma 1.1, 1.2, 1.9": partial(
                    gamma_function, exponent=[1.0 / 1.1, 1.0 / 1.2, 1.0 / 1.9]
                ),
            }
        )
        self.TF.CCTF_ENCODINGS.update(CUSTOM_ENCODINGS)
        self.functions = {
            "encode": self.TF.cctf_encoding,
            "decode": self.TF.cctf_decoding,
        }

    def set_coding(self, name: str, exponent):
        self.TF.CCTF_DECODINGS.update(
            CanonicalMapping({name: partial(gamma_function, exponent=exponent)})
        )
        self.TF.CCTF_ENCODINGS.update(
            CanonicalMapping({name: partial(gamma_function, exponent=1/exponent)})
        )

    def apply_CCTF(self, mode: MODE, cctf_type: str, image):
        return self.functions[mode](value=image, function=cctf_type)
