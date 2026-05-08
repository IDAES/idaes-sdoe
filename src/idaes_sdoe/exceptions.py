# #################################################################################
# idaes-sdoe is part of the IDAES integrated software platform, specifically
# the Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework).
#
# Copyright (c) 2018-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
# #################################################################################
class IdaesSdoeError(Exception):
    pass


class ConfigurationError(IdaesSdoeError):
    pass


class DataValidationError(IdaesSdoeError):
    pass


class OptionalDependencyError(IdaesSdoeError):
    pass


class UnsupportedFeatureError(IdaesSdoeError):
    pass
