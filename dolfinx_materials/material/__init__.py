#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

@Author  :   Jeremy Bleyer, Ecole Nationale des Ponts et Chaussées, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   08/06/2023
"""
from .generic import Material

try:
    from .mfront import MFrontMaterial
except ImportError:
    print("MGIS is not available. MFront behaviors cannot be used.")
