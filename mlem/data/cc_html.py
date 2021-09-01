
import datetime
import email.parser
import gzip
import itertools
import os

import pyarrow as pa
import s3fs
import tqdm
import warcio

from . import util

def list_warc_files(base, crawl, fs):
    with fs.open(f'{base}/crawl-data/{crawl}/warc.paths.gz') as rf:
        with gzip.open(rf, 'rt') as f:
            yield from map(lambda l: l.rstrip('\n'), f.readlines())

def scan_warc_files(src, base, fs):
    rectypes = set(('request', 'response', 'metadata'))
    def group():
        for path in src:
            with fs.open(f'{base}/{path}') as stream:
                evts = {}
                for rec in warcio.archiveiterator.ArchiveIterator(stream):
                    if rec.rec_type not in rectypes:
                        continue
                    hdr = rec.rec_headers
                    uri = hdr.get_header('WARC-Target-URI')
                    if uri not in evts:
                        evts[uri] = {}
                    evt = evts[uri]
                    # the content_stream is not coherent after the iter has
                    # moved on, so we need to grab it now.
                    evt[rec.rec_type] = (rec, rec.content_stream().read())
                    if all(t in evt for t in rectypes):
                        yield evt
                        del evts[uri]
                if len(evts):
                    raise(NotImplementedError('unpaired records'))
    for evt in group():
        req = evt.get('request')
        res = evt.get('response')
        dat = evt.get('metadata')
        if not util.seems_like_html(res[1]):
            continue
        body = util.body_text(*res)
        if "\0" in body:
            # null bytes in the middle of the stream are rare, but
            # problematic
            continue
        hdr = email.parser.BytesParser().parsebytes(dat[1])
        ftm = hdr.get('fetchTimeMs')
        yield {
            'html': body,
            'uri': req[0].rec_headers.get_header('WARC-Target-URI'),
            'req': req[0].http_headers.to_str() + "\r\n" + util.body_text(*req),
            'res': res[0].http_headers.to_str(),
            'when': datetime.datetime.strptime(
                req[0].rec_headers.get_header('WARC-Date'),
                '%Y-%m-%dT%H:%M:%SZ'),
            'duration': float(ftm) / 1000.0 if ftm else None,
        }

def get():
    dst = 'cc-html.parquet'
    if os.path.exists(dst):
        return dst    
    base = 's3://commoncrawl'
    crawl = 'CC-MAIN-2017-13'
    s3 = s3fs.S3FileSystem(anon=True)
    src = list_warc_files(base, crawl, s3)
    # 66500 warc files in this sample, first file contains 43944
    # usable responses, that's enough for the moment.  It's cool
    # that there are 2.9 billion available, but I don't have enough
    # sand & lightning to consume them right now.
    src = (next(src),) # just get the first warc file
    src = scan_warc_files(src, base, s3)
    src = tqdm.tqdm(src)
    util.toParquet(dst, src,
        schema=pa.schema((
            ('html', pa.string()),
            ('uri', pa.string()),
            ('req', pa.string()),
            ('res', pa.string()),
            ('when', pa.timestamp('s')),
            ('duration', pa.float64()),
        )),
    )
    return dst
