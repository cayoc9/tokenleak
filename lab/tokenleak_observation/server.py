"""Loopback-only laboratory service; server diagnostics are privileged data."""
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from pathlib import Path
import signal
import socket
import time

from .transport import FRAME_BYTES, marker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--condition', choices=['gpt2', 'gpt2_input_blinded', 'constant'], required=True)
    p.add_argument('--port-file', required=True, type=Path)
    p.add_argument('--trace', required=True, type=Path)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    model = None
    load_started = time.perf_counter_ns()
    if args.condition != 'constant':
        import torch
        from transformers import GPT2LMHeadModel
        torch.set_num_threads(cfg['torch_threads'])
        torch.set_num_interop_threads(1)
        torch.manual_seed(cfg['model_seed'])
        model = GPT2LMHeadModel.from_pretrained(str(args.model), local_files_only=True,
                    use_safetensors=True, attn_implementation='eager').eval()
    loaded_ns = time.perf_counter_ns() - load_started

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def do_GET(self):
            body = b'ok\n'
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers['Content-Length'])
            if length != cfg['request_body_bytes']:
                self.send_error(400)
                return
            request = json.loads(self.rfile.read(length))
            ids = list(request['input_ids'])
            steps = int(request['steps'])
            if len(ids) != cfg['input_length'] or steps != cfg['output_steps']:
                self.send_error(400)
                return
            if args.condition == 'gpt2_input_blinded':
                ids[cfg['secret_position']] = cfg['blinding_token_id']
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson')
            self.send_header('Content-Length', str(steps * FRAME_BYTES))
            self.end_headers()
            self.wfile.flush()
            times, outputs = [], []
            if model is None:
                for index in range(steps):
                    self.wfile.write(marker(index)); self.wfile.flush()
            else:
                with torch.inference_mode():
                    current = torch.tensor([ids], dtype=torch.long)
                    past = None
                    for index in range(steps):
                        begin = time.perf_counter_ns()
                        prediction = model(current, past_key_values=past, use_cache=True)
                        next_token = prediction.logits[:, -1, :].argmax(dim=-1)
                        past = prediction.past_key_values
                        times.append(time.perf_counter_ns() - begin)
                        outputs.append(int(next_token.item()))
                        self.wfile.write(marker(index)); self.wfile.flush()
                        current = next_token[:, None]
            trace = {'request_id': request['request_id'], 'purpose': request['purpose'],
                     'condition': args.condition, 'model_forward_ns': times,
                     'generated_token_ids': outputs, 'model_load_ns': loaded_ns,
                     'visibility': 'privileged diagnostic; excluded from attacker features'}
            with args.trace.open('a') as f:
                f.write(json.dumps(trace) + '\n')

    server = HTTPServer(('127.0.0.1', 0), Handler)
    args.port_file.write_text(json.dumps({'port': server.server_address[1], 'pid': __import__('os').getpid(),
                                       'bound_address': '127.0.0.1', 'model_load_ns': loaded_ns}) + '\n')
    server.serve_forever()


if __name__ == '__main__':
    main()
