#!/usr/bin/python3

"""
submit execution log
"""

import argparse
import errno
#import json
import os
import pwd
import re
import sqlite3
import sys
import tempfile
import time
import parse_log

default_data_dir = "/home/share/public_html/parallel-distributed/21mnist/records/mnist_records"
#default_data_dir = "mnist_records"

dbg = 0

# --------- nuts and bolts ---------

def Ws(msg):
    """
    write to stdout
    """
    sys.stdout.write(msg)
    sys.stdout.flush()

def Es(msg):
    """
    write to stderr
    """
    sys.stderr.write(msg)

def ensure_directory(directory):
    """
    ensure directory exists
    """
    try:
        os.makedirs(directory)
    except OSError as err:
        assert(err.errno == errno.EEXIST), err

# database nuts and bolts

def do_sql(con, cmd, dbg_level, *vals):
    """
    do sql statement.
    dbg_level is the level of verbosity above which
    it is printed
    """
    if dbg_level <= dbg:
        Es("%s with %s\n" % (cmd, vals))
    return con.execute(cmd, vals)

def read_schema(con):
    """
    read schema of all tables
    """
    col_cmd = 'select name, sql from sqlite_master where type = "table"'
    pat = re.compile(r"CREATE TABLE [^\(]+\((?P<cols>.*)\)")
    schema = {}
    for name, create_table in do_sql(con, col_cmd, 1):
        match = pat.match(create_table)
        assert(match), create_table
        cols = match.group("cols")
        schema[name] = [s.strip() for s in cols.split(",")]
    return schema

def ensure_columns(con, schema, tbl, columns):
    """
    ensure table TBL exists and has COLUMNS
    """
    if tbl not in schema:
        create_cmd = "create table if not exists {}({})".format(tbl, ",".join(columns))
        do_sql(con, create_cmd, 1)
        schema[tbl] = columns
    else:
        existing_columns = schema[tbl]
        for col in columns:
            if col not in existing_columns:
                alt_cmd = "alter table {} add {}".format(tbl, col)
                do_sql(con, alt_cmd, 1)
                existing_columns.append(col)
        schema[tbl] = existing_columns

def open_for_transaction(sqlite3_file):
    """
    open database for transaction
    """
    con = sqlite3.connect(sqlite3_file)
    con.row_factory = sqlite3.Row
    schema = read_schema(con)
    if "seq_counter" not in schema:
        do_sql(con, "create table seq_counter(x)", 1)
        do_sql(con, "insert into seq_counter(x) values(?)", 1, 0)
    schema = read_schema(con)
    return con, schema

def get_next_seqid(con):
    """
    return next seqid
    """
    [(seqid,)] = list(do_sql(con, "select x from seq_counter", 1))
    do_sql(con, "update seq_counter set x = ?", 1, seqid + 1)
    return seqid

def delete_from_db(con, schema, delete_seqids, delete_mine, user):
    """
    delete records of specified seqids from database
    """
    if "info" in schema:
        user_seqids_q = 'select distinct seqid from info where owner = ?'
        user_seqids = {row["seqid"] for row in do_sql(con, user_seqids_q, 1, user)}
        delete_seqids_q = ('select distinct seqid from info where seqid in ({})'
                           .format(",".join([str(x) for x in delete_seqids])))
        delete_seqids = {row["seqid"] for row in do_sql(con, delete_seqids_q, 1)}
    else:
        user_seqids = set()
        delete_seqids = set()
    diff = delete_seqids.difference(user_seqids)
    if len(diff) > 0:
        Es("warning: you can't delete seqids %s, as they are not yours\n"
           % sorted(list(diff)))
    if delete_mine:
        seqids = user_seqids
    else:
        seqids = delete_seqids.intersection(user_seqids)
    if len(seqids) > 0:
        seqids_comma = ",".join([("%d" % x) for x in sorted(list(seqids))])
        for tbl, _ in schema.items():
            if tbl != "seq_counter":
                cmd = "delete from %s where seqid in (%s)" % (tbl, seqids_comma)
                do_sql(con, cmd, 1)
    return seqids

def parse_val(x):
    """
    parse a string into an sqlite3 value
    """
    if x is None:
        return None
    try:
        return int(x)
    except ValueError:
        pass
    try:
        return float(x)
    except ValueError:
        pass
    return x

def insert_row(con, schema, tbl, row, seqid):
    """
    insert a row into database
    """
    row = {k.replace("-", "_") : v for k, v in row.items()}
    fields = ["seqid"] + list(row.keys())
    row["seqid"] = seqid
    n_fields = len(fields)
    ins_cmd = ("insert into {}({}) values({})"
               .format(tbl, ",".join(fields), ",".join(["?"] * n_fields)))
    vals = [row[f] for f in fields]
    ensure_columns(con, schema, tbl, fields)
    do_sql(con, ins_cmd, 2, *vals)
    return 1

def insert_rows(con, schema, tbl, rows, seqid):
    """
    insert rows into database
    """
    n_inserted = 0
    for row in rows:
        n_inserted += insert_row(con, schema, tbl, row, seqid)
    return n_inserted

def make_row_from_key_vals(rows):
    """
    rows is a list of {"key" : x, "val" y}.
    return a dictionary of {x : y, ...}
    along with the list of keys as they appeared
    in rows
    """
    dic = {}
    keys = [row["key"] for row in rows]
    for row in rows:
        dic[row["key"]] = row["val"]
    return keys, dic

def insert_into_db(con, schema, user, logs):
    """
    insert all records in plogs into database
    """
    seqids = []
    for parsed, _ in logs:
        seqid = get_next_seqid(con)
        seqids.append(seqid)
        for tbl, rows in parsed.items():
            if tbl == "key_vals":
                rows.append(dict(key="owner", val=user))
                _, dic = make_row_from_key_vals(rows)
                insert_row(con, schema, "info", dic, seqid)
            else:
                insert_rows(con, schema, tbl, rows, seqid)
    return seqids

def ensure_data_dir(data_dir):
    """
    ensure directories data_dir/{queue,commit,deleted} exist
    """
    queue_dir = "{}/queue".format(data_dir)
    commit_dir = "{}/commit".format(data_dir)
    deleted_dir = "{}/deleted".format(data_dir)
    ensure_directory(data_dir)
    ensure_directory(queue_dir)
    ensure_directory(commit_dir)
    ensure_directory(deleted_dir)
    return queue_dir, commit_dir, deleted_dir

# ------------------------------

def parse_delete_seqids(delete_seqids):
    """
    "1,2,3" --> [1,2,3]
    """
    if delete_seqids is None:
        return []
    ids = None
    try:
        ids = [int(x) for x in delete_seqids.split(",")]
    except ValueError:
        pass
    if ids is None:
        Es("argument to --delete-seqids must be N,N,...\n")
    return ids

def parse_logs(logs, q_dir):
    """
    parse all files in logs
    """
    result = []
    for log in logs:
        parsed, raw_data = parse_log.parse_log(log)
        if q_dir is None:
            q_log = None
        else:
            prefix = time.strftime("%Y-%m-%d-%H-%M-%S")
            tmp_fd, q_log = tempfile.mkstemp(suffix=".log", prefix=prefix, dir=q_dir)
            tmp_wp = os.fdopen(tmp_fd, "w")
            tmp_wp.write(raw_data)
            tmp_wp.close()
        result.append((parsed, q_log))
    return result

def move_to_dir(file, seqid, to_dir):
    """
    move FILE to TO_DIR/SEQID-FILE
    """
    _, orig_file = os.path.split(file)
    dest_file = "{}/{:06d}-{}".format(to_dir, seqid, orig_file)
    os.rename(file, dest_file)

def create_file(seqid, dire):
    """
    create an empty file in DIRE, to record SEQID is deleted
    """
    dest_file = "{}/{:06d}".format(dire, seqid)
    dest_wp = open(dest_file, "w")
    dest_wp.close()

def get_user():
    """
    get real user ID of the process
    """
    uid = os.getuid()
    return pwd.getpwuid(uid).pw_name

def get_euser():
    """
    get effective user ID of the process
    """
    uid = os.geteuid()
    return pwd.getpwuid(uid).pw_name

def parse_args(argv):
    """
    parse command line args
    """
    psr = argparse.ArgumentParser()
    psr.add_argument("files", metavar="FILE",
                     nargs="*", help="files to submit")
    psr.add_argument("--dryrun", "-n",
                     action="store_true", help="dry run")
    psr.add_argument("--pretend", "-p", metavar="USER",
                     action="store",
                     help=("pretend as if USER is executing this command."
                           "  Affect the owner field of inserted records"
                           " and records deleted by --delete-mine;"
                           " only the owner of the command can pretend other users"))
    psr.add_argument("--data", metavar="DIRECTORY",
                     default=default_data_dir, action="store",
                     help="generate database/csv under DIRECTORY")
    psr.add_argument("--delete-seqids", "-d", metavar="ID,ID,...",
                     action="store",
                     help="delete data of specified seqids")
    psr.add_argument("--delete-mine", "-D",
                     action="store_true",
                     help="delete all data of submitting user")
    psr.add_argument("--dbg", type=int, default=0,
                     help="specify debug level")
    opt = psr.parse_args(argv)
    if opt.pretend:
        user = get_user()
        euser = get_euser()
        if user not in [euser, opt.pretend]:
            Es("you ({}) cannot specify a different user with --pretend {}\n"
               .format(user, opt.pretend))
            return None
    else:
        opt.pretend = get_user()
    opt.delete_seqids = parse_delete_seqids(opt.delete_seqids)
    if opt.delete_seqids is None:
        return None
    if len(opt.delete_seqids) == 0 and (not opt.delete_mine) and len(opt.files) == 0:
        opt.files.append("-")
    global dbg
    dbg = opt.dbg
    return opt

def main():
    """
    main
    """
    args = parse_args(sys.argv[1:])
    if args is None:
        return 1
    logs = args.files[:]
    if args.dryrun:
        parse_logs(logs, None)
        Es("submit.py: dry run. do nothing\n")
        return 0
    data_dir = args.data
    q_dir, c_dir, d_dir = ensure_data_dir(data_dir)
    # parse and queue the contents
    q_logs = parse_logs(logs, q_dir)
    a_sqlite = "{}/a.sqlite".format(data_dir)
    con, schema = open_for_transaction(a_sqlite)
    deleted = delete_from_db(con, schema,
                             args.delete_seqids, args.delete_mine, args.pretend)
    inserted = insert_into_db(con, schema, args.pretend, q_logs)
    con.commit()
    con.close()
    for (_, q_log), seqid in zip(q_logs, inserted):
        move_to_dir(q_log, seqid, c_dir)
    for seqid in deleted:
        create_file(seqid, d_dir)
    if len(deleted) + len(inserted) > 0:
        Es("database {} updated ({} deleted, {} inserted)\n"
           .format(a_sqlite, len(deleted), len(inserted)))
    return 0

main()
