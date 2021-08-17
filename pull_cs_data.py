#!/usr/bin/env python3

import io
import multiprocessing as mp
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.fs as pf
import s3fs
import sniffpy
import tqdm
import warcio

from pull_data import body_text

def pull_ccidx(ds):
    for p in ds.pieces:
        for b in p.scanner(use_async=True, columns=[
            'url',
            'url_host_name',
            'warc_filename',
            'warc_record_offset',
            'warc_record_length',
        ]).to_batches():
            yield (pc.value_counts(b['url_host_name']),
                b.filter(pc.match_substring_regex(b['url'],
                    '^https?://[^/]+/?$')))

def unfold(d):
    ''' takes { a: [ 1, 2 ], b: [ 3, 4 ] }
      returns [ { a: 1, b: 3 }, { a: 2, b: 4 } ] '''
    ks = d.keys()
    for vs in zip(*d.values()):
        yield dict(zip(ks, vs))

def ccidx_join(src):
    # the task here is to aggregate entries that span batches
    # both with the counters and the index page entries
    prev = {}
    for vc, ix in src:
        blk = { _['values']: [ _['values'], _['counts'] ]
                for _ in vc.to_pylist() }
        for row in unfold(ix.to_pydict()):
            blk[row['url_host_name']].append(row)
        for k in prev.keys():
            if k in blk:
                blk[k] = [
                    k, prev[k][0] + blk[k][0]
                ] + prev[k][1:] + blk[k][1:]
            else: # no updates to k in this block, it's done
                yield prev[k]
        prev = blk
    yield from prev.values()

def warc_init(func):
    func.awrl = warcio.recordloader.ArcWarcRecordLoader()
    func.fs = s3fs.S3FileSystem(anon=True)

def warc_load(item):
    if len(item) < 3:
        return
    # just grab the first seen warc entry for now, the others are
    # likely duplicate responses
    host, count, row = item[:3]
    try:
        rec = warc_load.awrl.parse_record_stream(
            warcio.bufferedreaders.DecompressingBufferedReader(
                io.BytesIO(warc_load.fs.read_block(
                    'commoncrawl/' + row['warc_filename'],
                    offset=row['warc_record_offset'],
                    length=row['warc_record_length'],
                ))))
        body = rec.content_stream().read()
        mt = sniffpy.sniff(body)
        if mt.type != 'text' or mt.subtype != 'html':
            return
        body = body_text(rec, body)
        if "\0" in body:
            # null bytes in the middle of the stream are rare, but
            # problematic
            return
        return (body, count, host)
    except BaseException as e:
        print(e, file=sys.stderr)
        return

def warc_pull(src):
    with mp.Pool(len(os.sched_getaffinity(0)),
            initializer=warc_init, initargs=(warc_load,)) as p:
        yield from filter(lambda x: x is not None, p.imap(warc_load, src))

def write_pqt(dst, src, schema, limit=float('inf')):
    wr = pq.ParquetWriter(dst, schema, compression='GZIP')
    batch = []
    def flush():
        nonlocal batch
        rb = pa.table(list(map(pa.array, zip(*batch))), schema=schema)
        wr.write_table(rb)
        batch = []
    for i, row in enumerate(src):
        batch.append(row)
        if i+2 > limit:
            break
        if len(batch) >= 256:
            flush()
    flush()
    wr.close()

if __name__ == '__main__':
    src = pf.FileSystem.from_uri(
        # you can download some of the index parquets to speed up testing
        #'file:///commoncrawl/cc-index/table/cc-main/warc'
        's3://commoncrawl/cc-index/table/cc-main/warc'
    )
    src = pq.ParquetDataset(src[1],
        filters=[('crawl', '=', 'CC-MAIN-2017-13'), ('subset', '=', 'warc')],
        filesystem=src[0])
    src = pull_ccidx(src)
    src = ccidx_join(src)
    src = warc_pull(src)
    n = 180000
    src = tqdm.tqdm(src, total=n)
    write_pqt('cc-crawlsize.parquet', src, pa.schema((
        ('sentence', pa.string()),
        ('label', pa.float64()),
        ('hostname', pa.string()),
    )), limit=n)
