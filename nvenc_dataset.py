# MIT License
#
# Copyright (c) 2021 Keisuke Sehara
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

## the tuple of values indicate that the board has __either of__ these cores.

COMPAT = {
    # Kepler
    ## 600 series
    "GeForce GT 635":  ("GK208",),
    "GeForce GTX 645": ("GK106",),
    "GeForce GTX 650": ("GK107",),
    "GeForce GTX 650 Ti": ("GK106",),
    "GeForce GTX 650 Ti Boost": ("GK106",),
    "GeForce GTX 660": ("GK104", "GK106",),
    "GeForce GTX 660 Ti": ("GK104",),
    "GeForce GTX 670": ("GK104",),
    "GeForce GTX 680": ("GK104",),
    "GeForce GTX 690": ("GK104",),
    "GeForce GTX 660M": ("GK107",),
    "GeForce GTX 670MX": ("GK104",),
    "GeForce GTX 675MX": ("GK104",),
    "GeForce GTX 680M": ("GK104",),
    "GeForce GTX 680MX": ("GK104",),
    ## 700 series
    "GeForce GT 710": ("GK208",),
    "GeForce GT 720": ("GK208",),
    "GeForce GT 740": ("GK107",),
    "GeForce GTX 760 192-bit": ("GK104",),
    "GeForce GTX 760": ("GK104",),
    "GeForce GTX 760 Ti": ("GK104",),
    "GeForce GTX 770": ("GK104",),
    "GeForce GTX 780": ("GK110",),
    "GeForce GTX 780 Ti": ("GK110",),
    "GeForce GTX TITAN": ("GK110",),
    "GeForce GTX TITAN Black": ("GK110",),
    "GeForce GTX TITAN Z": ("GK110",),

    # Maxwell
    ## 700 series
    "GeForce GTX 745": ("GM107",),
    "GeForce GTX 750": ("GM107",),
    "GeForce GTX 750 Ti": ("GM107",),
    "GeForce GTX 760M": ("GK106",),
    "GeForce GTX 765M": ("GK106",),
    "GeForce GTX 770M": ("GK106",),
    "GeForce GTX 780M": ("GK104",),
    ## 800 series
    "GeForce GTX 850M": ("GM107",),
    "GeForce GTX 860M": ("GM107", "GK104",),
    "GeForce GTX 870M": ("GK104",),
    "GeForce GTX 880M": ("GK104",),
    ## 900 series
    "GeForce GTX 950": ("GM206",),
    "GeForce GTX 950 (OEM)": ("GM206",),
    "GeForce GTX 960": ("GM206",),
    "GeForce GTX 960 (OEM)": ("GM204",),
    "GeForce GTX 970": ("GM204",),
    "GeForce GTX 980": ("GM204",),
    "GeForce GTX 980 Ti": ("GM200",),
    "GeForce GTX TITAN X": ("GM200",),
    "GeForce GTX 950M": ("GM107",),
    "GeForce GTX 960M": ("GM107",),
    "GeForce GTX 965M": ("GM204",),
    "GeForce GTX 970M": ("GM204",),
    "GeForce GTX 980M": ("GM204",),

    # Pascal
    ## 10 series
    "GeForce GT 1010": ("GP108",),
    "GeForce GT 1030": ("GP108",),
    "GeForce GTX 1050": ("GP107",),
    "GeForce GTX 1050 Ti": ("GP107",),
    "GeForce GTX 1060": ("GP104-1xx", "GP106",),
    "GeForce GTX 1070": ("GP104-2xx+",),
    "GeForce GTX 1070 Ti": ("GP104-2xx+",),
    "GeForce GTX 1080": ("GP104-2xx+",),
    "GeForce GTX 1080 Ti": ("GP102",),
    "Nvidia TITAN X": ("GP102",),
    "Nvidia TITAN Xp": ("GP102",),
    "GeForce GTX 1050 (Notebook)": ("GP107",),
    "GeForce GTX 1050 Ti (Notebook)": ("GP107",),
    "GeForce GTX 1060 (Notebook)": ("GP106",),
    "GeForce GTX 1060 Max-Q": ("GP106",),
    "GeForce GTX 1070 (Notebook)": ("GP104",),
    "GeForce GTX 1070 Max-Q": ("GP104",),
    "GeForce GTX 1080 (Notebook)": ("GP104",),
    "GeForce GTX 1080 Max-Q": ("GP104",),

    # Volta
    "Nvidia TITAN V": ("GV10x",),
    "Nvidia TITAN V CEO Edition": ("GV10x",),

    # Turing
    ## 16 series
    "GeForce GTX 1650": ("TU117",),
    "GeForce GTX 1650 Super": ("TU116",),
    "GeForce GTX 1660": ("TU116",),
    "GeForce GTX 1660 Super": ("TU116",),
    "GeForce GTX 1660 Ti": ("TU116",),
    "GeForce GTX 1650 (Laptop)": ("TU117",),
    "GeForce GTX 1650 Max-Q": ("TU117",),
    "GeForce GTX 1650 Ti Max-Q": ("TU117",),
    "GeForce GTX 1650 Ti": ("TU117",),
    "GeForce GTX 1660 (Laptop)": ("TU116",),
    "GeForce GTX 1660 Ti Max-Q": ("TU116",),
    "GeForce GTX 1660 Ti (Laptop)": ("TU116",),
    ## 20 series
    "GeForce RTX 2060": ("TU104", "TU106",),
    "GeForce RTX 2060 Super": ("TU106",),
    "GeForce RTX 2070": ("TU106",),
    "GeForce RTX 2070 Super": ("TU104",),
    "GeForce RTX 2080": ("TU104",),
    "GeForce RTX 2080 Super": ("TU104",),
    "GeForce RTX 2080 Ti": ("TU102",),
    "Nvidia TITAN RTX": ("TU102",),
    "GeForce RTX 2060 (Laptop)": ("TU106",),
    "GeForce RTX 2060 Max-Q": ("TU106",),
    "GeForce RTX 2070 (Laptop)": ("TU106",),
    "GeForce RTX 2070 Max-Q": ("TU106",),
    "GeForce RTX 2070 Super (Laptop)": ("TU104",),
    "GeForce RTX 2070 Super Max-Q": ("TU104",),
    "GeForce RTX 2080 (Laptop)": ("TU104",),
    "GeForce RTX 2080 Max-Q": ("TU104",),
    "GeForce RTX 280 Super (Laptop)": ("TU104",),
    "GeForce RTX 2080 Super Max-Q": ("TU104",),

    # Ampere
    ## 30 series
    "GeForce RTX 3060": ("GA106", "GA104",),
    "GeForce RTX 3060 Ti": ("GA104",),
    "GeForce RTX 3070": ("GA104",),
    "GeForce RTX 3070 Ti": ("GA104",),
    "GeForce RTX 3080": ("GA102",),
    "GeForce RTX 3080 Ti": ("GA102",),
    "GeForce RTX 3090": ("GA102",),
    "GeForce RTX 3050": ("GA107",),
    "GeForce RTX 3050 Ti": ("GA107",),
    "GeForce RTX 3060 (Laptop)": ("GA106",),
    "GeForce RTX 3070 (Laptop)": ("GA104",),
    "GeForce RTX 3080 (Laptop)": ("GA104",),
}

AMBIG  = {
    # 600 series
    "GeForce GT 630":  ("GK107", "GF108", "GK208"),
    "GeForce GT 640":  ("GF116", "GK107", "GK208"),
    "GeForce GT 640M": ("GK107",),
    "GeForce GT 645M": ("GK107",),
    "GeForce GT 650M": ("GK107",),
    # 700 series
    ## Wikipedia says 'mobile no-GTX cores lack support NVENC'
    "GeForce GT 730":  ("GK208", "GF108"),
    "GeForce GT 720M": ("GF117", "GK208"),
    "GeForce GT 730M": ("GK208",),
    "GeForce GT 735M": ("GK208",),
    "GeForce GT 740M": ("GK208", "GK107",),
    "GeForce GT 745M": ("GK107",),
    "GeForce GT 750M": ("GK107",),
    "GeForce GT 755M": ("GK107",),
    # 900 series
    ## Wikipedia says 'mobile no-GTX cores lack support NVENC'
    "GeForce 940MX": ("GM108", "GM107",),
}

NONE   = {
    # 600 series
    "GeForce GT 645": ("GF114",),
    "GeForce 610M": ("GF119",),
    "GeForce GT 620M": ("GF117",),
    "GeForce GT 625M": ("GF117",),
    "GeForce GT 630M": ("GF108",),
    "GeForce GT 635M": ("GF106",),
    "GeForce GT 640M LE": ("GF108", "GK107",),
    "GeForce GTX 670M": ("GF114",),
    "GeForce GTX 675M": ("GF114",),

    # 700 series
    "GeForce GT 705": ("GF119",),
    "GeForce 710M": ("GF117",),

    # 800 series
    ## Wikipedia says 'mobile no-GTX cores lack support NVENC'
    "GeForce 810M": ("GF117",),
    "GeForce 820M": ("GF117",),
    "GeForce 825M": ("GK208",),
    "GeForce 830M": ("GM108",),
    "GeForce 840M": ("GM108",),
    "GeForce 845M": ("GM108", "GM107",),

    # 900 series
    ## Wikipedia says 'mobile no-GTX cores lack support NVENC'
    "GeForce GT 945A": ("GM108",), # exception
    "GeForce 920M": ("GK208",),
    "GeForce 930M": ("GM108",),
    "GeForce 940M": ("GM108",),

    # 10 series / 20 series
    ## Wikipedia says 'MX cores lack support NVENC'
    "GeForce MX110": ("GM108",),
    "GeForce MX130": ("GM108",),
    "GeForce MX150": ("GP108",),
    "GeForce MX230": ("GP108",),
    "GeForce MX250": ("GP108",),
    "GeForce MX330": ("GP108",),
    "GeForce MX350": ("GP107",),
    "GeForce MX450": ("TU117",),
}

NVENC_CORES_H264 = {
    "GK110": 1,
    "GK107": 1,
    "GK106": 1,
    "GK104": 1,
    "GM108": 0,
    "GM107": 1,
    "GM208": 1,
    "GM206": 1,
    "GM204": 2,
    "GM200": 2,
    "GP108": 0,
    "GP107": 1,
    "GP106": 1,
    "GP104-2xx+": 2,
    "GP104-1xx": 1,
    "GP102": 2,
    "GP100": 3,
    "GV10x": 3,
    "TU117": 1,
    "TU116": 1,
    "TU106": 1,
    "TU104": 1,
    "TU102": 1,
    "GA106": 1,
    "GA104": 1,
    "GA102": 1,
    "GA100": 0,
}
