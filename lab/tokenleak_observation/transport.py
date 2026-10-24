"""Observe actual socket reads; a read may complete several token markers."""
from collections import Counter
import json
import socket
import time

FRAME_BYTES = 64


def marker(index):
    body = json.dumps({'i': index}, separators=(',', ':')).encode()
    if len(body) >= FRAME_BYTES:
        raise ValueError('Marker exceeds fixed frame size')
    return body + b' ' * (FRAME_BYTES - len(body) - 1) + b'\n'


class Observer:
    def __init__(self, steps):
        self.steps = steps
        self.header = b''
        self.buffer = b''
        self.header_done = False
        self.reads = []
        self.frames = []
        self.body_bytes = 0

    def feed(self, raw, elapsed_ns):
        read_id = len(self.reads)
        body = raw
        if not self.header_done:
            self.header += raw
            if b'\r\n\r\n' not in self.header:
                self.reads.append({'read_id': read_id, 'elapsed_ns': elapsed_ns, 'bytes': len(raw), 'body_bytes': 0})
                return
            header, body = self.header.split(b'\r\n\r\n', 1)
            if not header.startswith(b'HTTP/1.0 200'):
                raise ValueError('Unexpected server response')
            self.header_done = True
            self.header = b''
        self.reads.append({'read_id': read_id, 'elapsed_ns': elapsed_ns, 'bytes': len(raw), 'body_bytes': len(body)})
        self.body_bytes += len(body)
        self.buffer += body
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            if len(line) + 1 != FRAME_BYTES:
                raise ValueError('Unexpected marker length')
            event = json.loads(line)
            if set(event) != {'i'} or event['i'] != len(self.frames):
                raise ValueError('Invalid marker sequence')
            self.frames.append({'index': event['i'], 'elapsed_ns': elapsed_ns, 'read_id': read_id})

    def finish(self):
        if len(self.frames) != self.steps or self.buffer or self.body_bytes != self.steps * FRAME_BYTES:
            raise ValueError('Incomplete response: no positions silently dropped')
        grouping = Counter(x['read_id'] for x in self.frames)
        return {'reads': self.reads, 'markers': self.frames, 'body_bytes': self.body_bytes,
                'body_reads': sum(r['body_bytes'] > 0 for r in self.reads),
                'max_markers_per_read': max(grouping.values()), 'ok': True}


def observe(port, request, steps, body_bytes=2048):
    body = json.dumps(request, separators=(',', ':')).encode()
    if len(body) > body_bytes:
        raise ValueError('Request exceeds fixed padding budget')
    body += b' ' * (body_bytes - len(body))
    headers = (f'POST /generate HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n'
               f'Content-Length: {len(body)}\r\nConnection: close\r\n\r\n').encode()
    parser = Observer(steps)
    with socket.create_connection(('127.0.0.1', int(port)), timeout=60) as connection:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        started = time.perf_counter_ns()
        connection.sendall(headers + body)
        while True:
            raw = connection.recv(4096)
            elapsed = time.perf_counter_ns() - started
            if not raw:
                break
            parser.feed(raw, elapsed)
    result = parser.finish()
    result['request_body_bytes'] = len(body)
    return result
