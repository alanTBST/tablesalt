"""
TBST Trafik, Bygge, og Bolig -styrelsen

Package for revenue distribution
based on pendler kombi rejsekort

The goal is to use pendler kombi rejsekort users
as a proxy for the more traditional pendler cards
to determine the average zone shares for operators
for each zone combination that pendler users may have


Author: Alan Jones alkj@tbst.dk, alanksjones@gmail.com

"""

from .connections import (make_connection,
                          check_dw_for_table,
                          insert_query, make_store)
from . import triptools

from . import io


