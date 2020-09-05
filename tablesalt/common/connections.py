"""
TBST Trafik, Bygge, og Bolig -styrelsen


Author: Alan Jones alkj@tbst.dk, alanksjones@gmail.com

"""

import pkg_resources
import json


import lmdb
import msgpack
import pandas as pd

try:
    from turbodbc import connect, make_options
    _MOD = 'turbodbc'
except ImportError:
    from pyodbc import connect
    _MOD = 'pyodbc'

def _load_connection_config():
    """Get db conn info."""
    conf_fp = pkg_resources.resource_filename(
        'tablesalt',
        'resources/config/db_connection.json'
        )
    with open(conf_fp, 'r') as fp:
        asdict = json.load(fp)
    return asdict





def make_connection():
    """
    Make a connection to the datawarehouse.

    Returns
    -------
    Connection object
        a turbodbc/pyodbc connection object with the info
        in CNXN_CONF

    """
    CNXN_CONF = _load_connection_config()
    if _MOD == 'pyodbc':
        url = ''
        for k, v in CNXN_CONF.items():
            url += k + '=' + v + ';'
        url = url.rstrip(';')
        return connect(url)

    ops = make_options(prefer_unicode=True,
                       use_async_io=True)

    return connect(driver=CNXN_CONF['DRIVER'],
                   server=CNXN_CONF['SERVER'],
                   database=CNXN_CONF['DATABASE'],
                   turbodbc_options=ops)


def check_dw_for_table(table_name, schema='Rejseplanen'):
    """
    Check if a table exists in the datawarehouse.

    Parameters
    ----------
    table_name : str
        the name of the table you're looking for
    schema : str, optional
        the schema for the table in the database
        The default is 'Rejseplanen'
    Returns
    -------
    bool
        True if the rejseplan stops are in the
        datawarehouse, False otherwise.

    """
    query = ("SELECT * FROM INFORMATION_SCHEMA.TABLES "
             f"WHERE TABLE_SCHEMA = '{schema}'")
    with make_connection() as con:
        tables = pd.read_sql(query, con)

    return table_name in set(tables.loc[:, 'TABLE_NAME'])


def insert_query(schema, table_name, ncols):

    vals = ''
    for n in range(ncols):
        vals += '?, '
    vals = vals.rstrip(', ')
    vals = '(' + vals + ')'

    return (f"INSERT INTO [{schema}].[{table_name}] "
            f"VALUES {vals}")

def _to_bytes(val):
    if not isinstance(val, bytes):
        if not isinstance(val, str):
            val = str(val)
        val = bytes(val, 'utf-8')

    return val

def to_bytes_msg(val):

    return msgpack.packb(val)


def _make_key_val(d, db_path, map_size=None):
    """
    Create a lmdb key-value store

    Parameters
    ----------
    d : dict
        the dictionary to store.
    db_path : str/path
        path to the store to create.
    map_size : int, optional
        The memeory mappping size of the datastore.
        The default is None. No data is written

    Returns
    -------
    None.

    """
    env = lmdb.open(db_path, map_size=map_size) # 18 gb.
    with env.begin(write=True) as txn:
        for k, v in d.items():
            if not isinstance(v, bytes):
                if not isinstance(v, str):
                    v = str(v)
                v = bytes(v, 'utf-8')
            if not isinstance(k, bytes):
                if not isinstance(k, str):
                    k = str(k)
                k = bytes(k, 'utf-8')
            txn.put(k, v)
        # print(env.stat()['entries'])
    env.close()

def make_store(d, db_path, start_size=1, size_limit=30):
    """
    Create an lmdb database

    Parameters
    ----------
    d : dict
        DESCRIPTION.
    db_path : str/path
        the path location of the store to create.
    start_size : int, optional
        The starting size of the datastore in GB. The default is 1.
    size_limit : int, optional
        The maximum size to try and make the key-value store

    Raises
    ------
    ValueError
        raise a ValueError if a store cannot be created with the
        give limit set.

    Returns
    -------
    None.

    """
    MAP_SIZES = {x: x * 1024 *1024 * 1024 for x in range(size_limit + 1)}
    i = start_size
    while True:
        try:
            _make_key_val(d, db_path, map_size=MAP_SIZES[i])
            break
        except Exception as e:
            print(str(e))
            i += 1
        if i >= size_limit:
            raise ValueError(
                f"failed to make key-value store: "
                f"size {db_path} > {size_limit}gb limit"
                )



