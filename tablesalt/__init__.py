"""

TABLESALT
=========

TBST Trafik, Bygge, og Bolig -styrelsen

Tablesalt is a package for analysing Rejsekort Delrejser data.
Delrejser, or partial trips, is a dataset created at Rejsekort that
contains a record for each check-in or checkout at a blue dot terminal.

This delrejser dataset is not the raw data recorded at a terminal in a station
or on bus, it is processed by Rejsekort to form "trips".

The rejsekort system itself records the sequence of transactions at terminals
and associated timestamps for the purposes of revenue collection and follows
the Danish public transport fare system. It is not designed with trip analysis
in mind.

For Example
-----------

If you purhase a 2 zone cash ticket from a machine or a kiosk in Sjælland,
that ticket is valid for 75 minutes. It is not only valid for a single
journey, but for any number of trips you can make in those 75 minutes,
provided you do not go further than two zones away from the zone you start
in. The rejsekort system mimics these conditions with the caveat that you are
to check-in when you start your journey, when you transfer transit modes and
checkout when your journey is finished.

If you're travelling using a rejsekort and you travel more than two zones
the system recognises you have and charges you accordingly. For each zone
distance the validity time increases and should you not travel further away
from you destination, you can make any number of trips within the time period
given by:

+---------+-----------+
|   zones |   minutes |
+=========+===========+
|       2 |        75 |
+---------+-----------+
|       3 |        90 |
+---------+-----------+
|       4 |       105 |
+---------+-----------+
|       5 |       120 |
+---------+-----------+
|       6 |       135 |
+---------+-----------+
|       7 |       150 |
+---------+-----------+
|       8 |       165 |
+---------+-----------+


The processed data groups all of these transactions for a user using these
zone distance and temporal validty thresholds and assigns a unique tripkey.
That definition of a trip is not necessarily equate to a transport analyst's
definition of a trip, be that trip-based, tour-based, activity-based etc.

This is where tablesalt aims to help out by being able to split trips up
and validate operators and modes.

The delrejser data produced by Rejsekort, while comprehensive still contains
errors in leg operators and timestamp/sequence mismatches.
Tablesalt provides functions and classes to tackle these errors.

Given that the rejsekort system relies on users actually checking in and out
where appropriate, the delrejser data contains incomplete trips. Methods are
provided to deal with these as well.

Using tablesalt you are also able to produce various origin-destination matrices of
rejsekort trips, as well as find operator zone work per trip and to model
other types of tickets


:Author: Alan Jones alkj@tbst.dk, alanksjones@gmail.com

"""
from . import common
from . import topology
from . import revenue
from . import season

from .common.io import StoreReader
from . import running

from ._version import get_versions
__version__ = get_versions()['version']
del get_get_versions

