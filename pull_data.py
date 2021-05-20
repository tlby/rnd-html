#!/usr/bin/env python3

# download strategy liberally plagiarized from https://github.com/commoncrawl/cc-pyspark/blob/9b1fe7955caa9bea99d36829f75a5fcbd71f0d9a/get-data.sh
import cgi
import codecs
import datetime
import email.parser
import gzip
import io
import json
import itertools
import re

#import chardet
import html5prescan
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import requests_cache
requests_cache.install_cache('crawl-data')
import tqdm
import warcio

class IterAsFile(io.TextIOBase):
    ''' it kind of blows my mind I had to write this adapter '''
    def __init__(self, iterable):
        self._iter = itertools.chain.from_iterable(iterable)
    def read(self, n=None):
        return bytearray(itertools.islice(self._iter, None, n))

def list_warc_files(base_url, crawl):
    r = requests.get(f'{base_url}/crawl-data/{crawl}/warc.paths.gz')
    stream = IterAsFile(r.iter_content(chunk_size=2**16))
    with gzip.open(stream, 'rt') as f:
        return map(lambda l: '%s/%s' % (base_url, l.rstrip('\n')),
            f.readlines())

def body_text(rec, src):
    ''' https://html.spec.whatwg.org/multipage/parsing.html#determining-the-character-encoding '''
    # 1) BOM sniffing
    def detect_bom(data, rec):
        if data.startswith(b'\xEF\xBB\xBF'):
            return 'UTF-8'
        if data.startswith(b'\xFE\xFF'):
            return 'UTF-16BE'
        if data.startswith(b'\xFF\xFE'):
            return 'UTF-16LE'
        # this isn't part of the official algorithm, but according to
        # https://en.wikipedia.org/wiki/Charset_detection
        # > One of the few cases where charset detection works reliably
        # > is detecting UTF-8. This is due to the large percentage of
        # > invalid byte sequences in UTF-8, so that text in any other
        # > encoding that uses bytes with the high bit set is extremely
        # > unlikely to pass a UTF-8 validity test.
        # so, we should try utf8 early and if it works, call that good
        return 'UTF-8'
    # 2) user-override (not applicable)
    # 3) wait for more bytes (already here)
    # 4) transport layer
    def detect_ctype(data, rec):
        if rec.http_headers:
            return cgi.parse_header(rec.http_headers.get(
                'content-type'))[1].get('charset')
    # 5) prescan
    def detect_pscan(data, rec):
        return html5prescan.get(data)[0].pyname
    # 6) frame parent inheritance (not applicable)
    # 7) previous visits (not applicable)
    # 8) autodetect
    def detect_csdet(data, rec):
        # this one has been removed, it was only triggering on 0.72% of
        # samples and was not a strong performer over the prescan
        # method, and is much slower than other methods
        return chardet.detect(data).get('encoding')
    # 9) guess based on region (impractical)

    data = src
    seen = ()
    for detect in (detect_bom, detect_pscan, detect_ctype):
        try:
            enc = detect(data, rec)
            enc = codecs.lookup(enc).name # normalize encoding name
            seen += (enc,)
            if enc not in seen:
                seen |= set((enc,))
                return str(data, enc)
        except TypeError: # None
            seen += (None,)
        except LookupError: # unknown charset
            seen += (enc,)
        except UnicodeError:
            pass
    # fallback, minimize encoding failures
    def lossy(enc):
        try:
            txt = str(data, enc, 'replace')
            return(txt, txt.count("�"), enc)
        except (TypeError, LookupError):
            return('', float('inf'), enc)
    win = min(map(lossy, set(seen)), key=lambda t: t[1])
    if False and rec.rec_type == 'response':
        # Dump the success rate for each prediction method for running
        # studies on this algorithm
        print(json.dumps(seen + (
            win[2], win[1], len(data),
            rec.http_headers.get('content-type') if rec.http_headers else None,
        )))
    return win[0]

def scan_warc_file(src):
    rectypes = set(('request', 'response', 'metadata'))
    def group():
        for url in src:
            r = requests.get(url)
            stream = IterAsFile(r.iter_content(chunk_size=2**16))
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

    def flat_msg(t):
        rec, body = t
        m = rec.http_headers
        return "\r\n".join(
            (f"{m.protocol} {m.statusline}",) +
            tuple(f"{k}: {v}" for k, v in m.headers) +
            ('', body_text(rec, body),)
        )

    for evt in group():
        req = evt.get('request')
        res = evt.get('response')
        dat = evt.get('metadata')
        ct = res[0].http_headers.get('content-type')
        if ct and re.match(r'(?i)text/html\b', ct):
            hdr = email.parser.BytesParser().parsebytes(dat[1])
            ftm = hdr.get('fetchTimeMs')
            yield {
                'uri': req[0].rec_headers.get_header('WARC-Target-URI'),
                'req': flat_msg(req),
                'res': flat_msg(res),
                'when': datetime.datetime.strptime(
                    req[0].rec_headers.get_header('WARC-Date'),
                    '%Y-%m-%dT%H:%M:%SZ'),
                'duration': float(ftm) / 1000.0 if ftm else None,
            }

def batch(src, n):
    it = iter(src)
    while True:
        hunk = tuple(itertools.islice(it, n))
        if not hunk:
            break
        yield hunk

def toParq(dst, src, schema):
    sty = pa.struct(zip(schema.names, schema.types))
    with pq.ParquetWriter(dst, schema, compression='GZIP') as w:
        for rows in src:
            cols = pa.array(rows, sty).flatten()
            w.write_table(pa.Table.from_arrays(cols, schema=schema))

if __name__ == '__main__':
    src = list(list_warc_files(
        base_url='https://commoncrawl.s3.amazonaws.com',
        crawl='CC-MAIN-2017-13',
    ))
    src = scan_warc_file((src[0],))
    src = tqdm.tqdm(src)
    src = batch(src, 128)
    toParq('cc-html.parquet', src, pa.schema((
        ('uri', pa.string()),
        ('req', pa.string()),
        ('res', pa.string()),
        ('when', pa.timestamp('s')),
        ('duration', pa.float64()),
    )))
