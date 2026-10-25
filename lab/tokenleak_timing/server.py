"""Loopback victim for v2; GPU completion occurs before emitting each fixed marker."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import socket
import time

from tokenleak_observation.transport import FRAME_BYTES, marker
from .contract import dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--condition', choices=['model', 'input_blinded', 'constant'], required=True)
    parser.add_argument('--port-file', required=True, type=Path)
    parser.add_argument('--trace', required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    settings = cfg['execution']
    model = None
    runtime = {'condition': args.condition, 'inference': False}
    if args.condition != 'constant':
        import torch
        import transformers
        from transformers import GPT2LMHeadModel
        torch.set_num_threads(settings['threads'])
        torch.set_num_interop_threads(1)
        torch.manual_seed(cfg['model']['seed'])
        if settings['device'].startswith('cuda:'):
            if not torch.cuda.is_available():
                raise ValueError('Requested CUDA device unavailable; no CPU fallback')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        dtype = getattr(torch, settings['dtype'])
        model = GPT2LMHeadModel.from_pretrained(
            str(args.model), local_files_only=True, use_safetensors=True,
            trust_remote_code=False, torch_dtype=dtype, attn_implementation='eager').eval()
        model = model.to(settings['device'])
        if model.config.model_type != 'gpt2' or model.config.vocab_size != 50257:
            raise ValueError('Unexpected model architecture/vocabulary')
        runtime = {'condition': args.condition, 'inference': True, 'torch': torch.__version__,
                   'transformers': transformers.__version__, 'device': settings['device'],
                   'dtype': str(next(model.parameters()).dtype), 'torch_cuda_build': torch.version.cuda,
                   'model_parameters': sum(p.numel() for p in model.parameters()),
                   'completion_barrier': 'argmax scalar .item() before response marker; host-visible duration',
                   'diagnostics_are_gpu_kernel_time': False,
                   'tf32_allowed': False if settings['device'].startswith('cuda:') else None}
        if settings['device'].startswith('cuda:'):
            runtime['gpu_name'] = torch.cuda.get_device_name(settings['device'])

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
            if self.headers.get('Content-Length') != str(settings['request_bytes']):
                self.send_error(400); return
            request = json.loads(self.rfile.read(settings['request_bytes']))
            ids = request['input_ids']
            steps = request['steps']
            if (not isinstance(ids, list) or not 1 <= len(ids) <= 64
                    or any(type(x) is not int or not 0 <= x < 50257 for x in ids)
                    or type(steps) is not int or steps != settings['output_steps']):
                self.send_error(400); return
            ids = list(ids)
            if args.condition == 'input_blinded':
                position = cfg['design']['secret_position']
                if cfg['task'] != 'single_slot_ttft' or len(ids) != cfg['design']['input_length']:
                    self.send_error(400); return
                ids[position] = cfg['design']['blinding_token_id']
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson')
            self.send_header('Content-Length', str(steps * FRAME_BYTES))
            self.end_headers()
            self.wfile.flush()
            durations = []
            if model is None:
                for step in range(steps):
                    self.wfile.write(marker(step)); self.wfile.flush()
            else:
                with torch.inference_mode():
                    current = torch.tensor([ids], dtype=torch.long, device=settings['device'])
                    past = None
                    for step in range(steps):
                        started = time.perf_counter_ns()
                        result = model(current, past_key_values=past, use_cache=True)
                        next_token = result.logits[:, -1, :].argmax(dim=-1)
                        # .item() waits for this scalar; the marker cannot precede model completion.
                        int(next_token.item())
                        durations.append(time.perf_counter_ns() - started)
                        past = result.past_key_values
                        self.wfile.write(marker(step)); self.wfile.flush()
                        current = next_token[:, None]
            diagnostic = {'request_id': request['request_id'], 'purpose': request['purpose'],
                          'host_visible_forward_and_argmax_ns': durations,
                          'visibility': 'privileged diagnostic; excluded from ranking',
                          'phase_order': 'prefill then autoregressive decode; no input-position alignment'}
            with args.trace.open('a') as stream:
                stream.write(json.dumps(diagnostic) + '\n')

    server = HTTPServer(('127.0.0.1', 0), Handler)
    dump(args.port_file, {'port': server.server_address[1], 'bound_address': '127.0.0.1',
                         'runtime': runtime})
    server.serve_forever()


if __name__ == '__main__':
    main()
