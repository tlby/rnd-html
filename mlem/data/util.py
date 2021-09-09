
import cgi
import codecs
import io
import sys

import html5prescan
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
import sniffpy
import warcio

def toParquet(dst, src, schema, compression='GZIP', **kwds):
    def bundle(src):
        rows = []
        for row in src:
            rows.append(row)
            if len(rows) < 256:
                continue
            yield rows
            rows = []
        if len(rows):
            yield rows
    sty = pa.struct(zip(schema.names, schema.types))
    with pq.ParquetWriter(dst, schema, compression=compression, **kwds) as wr:
        for rows in bundle(src):
            cols = pa.array(rows, sty).flatten()
            wr.write_table(pa.Table.from_arrays(cols, schema=schema))

def seems_like_html(buf):
    ''' https://mimesniff.spec.whatwg.org/ '''
    try:
        mt = sniffpy.sniff(buf)
        return bool(mt.type == 'text' and mt.subtype == 'html')
    except TypeError as e:
        # see codeprentice-org/sniffpy#42
        # TypeError: 'int' object is not subscriptable
        pass
    return False

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
                # for now we're letting the minimizer decide, because the
                # enabled detectors and encoding attempts are all cheap
                #return str(data, enc)
        except TypeError: # None
            seen += (None,)
        except LookupError: # unknown charset
            seen += (enc,)
        except UnicodeError:
            pass
    # fallback, compare candidates to minimize obvious encoding issues
    def lossy(enc):
        try:
            txt = str(data, enc, 'replace')
            # more characters is obvious mojibake
            # more replacement characters is a secondary heuristic
            # (cheap hack here to combine primary and secondary int
            # comparisons into one float comparison because lazy)
            score = 1.0 * len(txt) * (1 - 2 / (1 + txt.count("�")))
            return(txt, score, enc)
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
    return win[0].rstrip("\0")

class CommonCrawl:
    def __init__(self, base='commoncrawl'):
        self.base = base
        self.awrl = warcio.recordloader.ArcWarcRecordLoader()
        self.fs = s3fs.S3FileSystem(anon=True)
    def load(self, fn, offset, length):
        return self.awrl.parse_record_stream(
            warcio.bufferedreaders.DecompressingBufferedReader(io.BytesIO(
                self.fs.read_block(fn=f'{self.base}/{fn}', offset=offset,
                    length=length))))
