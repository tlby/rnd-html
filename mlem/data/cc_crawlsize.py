
import collections
#import concurrent.futures
import io
import itertools
import os
import multiprocessing
import sys

import util

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pf
import pyarrow.parquet as pq
import tqdm

def pull_ccidx(src):
    for p in src:
        for b in p.scanner(use_async=True, columns=[
            'url',
            'url_host_name',
            'warc_filename',
            'warc_record_offset',
            'warc_record_length',
        ]).to_batches():
            yield (pc.value_counts(b['url_host_name']),
                b.filter(pc.match_substring_regex(b['url'],
                    # index pages served via https only
                    '^https?://[^/]+/?$')))
def ccidx_fold(src):
    # the batches will be ordered, BUT a single host can span one or
    # more batches.  To aggregate we need to only announce hosts from
    # the last batch that have not been update in this batch (and might
    # be updated again in the next batch).
    def fold(a, b):
        return (a[0], a[1] + b[1]) + a[2:] + b[2:]
    def unfold(ix):
        d = ix.to_pydict()
        ks = d.keys()
        return (dict(zip(ks, vs)) for vs in zip(*d.values()))
    blk = ({},)
    for vc, ix in src:
        blk += ({ _['values']: ( _['values'], _['counts'] )
                    for _ in vc.to_pylist() },)
        for row in unfold(ix):
            blk[1][row['url_host_name']] += (row,)
        for k in blk[0].keys():
            if k in blk[1]:
                blk[1][k] = fold(blk[0][k], blk[1][k])
            elif len(blk[0][k]) > 2: # only if we found an index page
                yield blk[0][k]
        blk = (blk[1],)
    yield from blk[0].values()

def warc_init(func):
    func.cc = util.CommonCrawl()

def warc_load(item):
    # there might be more than one warc entry, but we'll just take the
    # first right now and assume the others are basically near dups
    host, count, row = item[:3]
    rec = warc_load.cc.load(fn=row['warc_filename'],
        offset=row['warc_record_offset'], length=row['warc_record_length'])
    body = rec.content_stream().read()
    if util.seems_like_html(body):
        body = util.body_text(rec, body)
        if "\0" not in body:
            # toss the few documents that have null bytes mid stream
            return (body, count, host)

def warc_pull(src):
    #with concurrent.futures.ProcessPoolExecutor(
    #    initializer=warc_init, initargs=(warc_load,)) as p:
    with multiprocessing.Pool(len(os.sched_getaffinity(0)),
            initializer=warc_init, initargs=(warc_load,)) as p:
        yield from filter(lambda x: x is not None, p.imap(warc_load, src))

def get_data(dst):
    fs, path = pf.FileSystem.from_uri(
        # you can download some of the index parquets to speed up testing
        #f'file:///{os.getcwd()}/commoncrawl/cc-index/table/cc-main/warc'
        's3://commoncrawl/cc-index/table/cc-main/warc'
    )
    src = pq.ParquetDataset(path,
        filters=(
            #('crawl', '=', 'CC-MAIN-2017-13'),
            ('crawl', '=', 'CC-MAIN-2019-47'),
            ('subset', '=', 'warc'),
        ),
        filesystem=fs,
    ).fragments
    src = pull_ccidx(src)
    src = ccidx_fold(src)
    src = warc_pull(src)
    n = 180000
    src = itertools.islice(src, n)
    src = tqdm.tqdm(src, total=n)
    util.toParquet(dst, src, pa.schema((
        ('sentence', pa.string()),
        ('label', pa.float64()),
        ('hostname', pa.string()),
    )))

if __name__ == '__main__':
    dst = 'cc-crawlsize.parquet'
    if not os.path.exists(dst):
        get_data(dst)
