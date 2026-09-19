#!/usr/bin/env python3
"""Manual Qwen Flash Next MTP/cache regression checks against an isolated server.

Start the workstation Q6 configuration on a test port with a separate SSD directory.
Run --phase exercise; run --phase prepare-restart, restart that backend, then run
--phase after-restart with the same --state path. Requires requests; no user data.
"""
import argparse
import base64
import json
import random
import struct
import time
import uuid
import zlib
from pathlib import Path

import requests

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--url', default='http://127.0.0.1:18087')
ap.add_argument('--phase', choices=['exercise', 'parity', 'prepare-restart', 'after-restart'], default='exercise')
ap.add_argument('--state', type=Path, default=Path('/tmp/qwen-cache-live-state.json'))
ap.add_argument('--report', type=Path, default=Path('/tmp/qwen-cache-live-report.json'))
args = ap.parse_args()
results = []
session = 'cache-review-' + uuid.uuid4().hex


def post(path, payload, timeout=120):
    r = requests.post(args.url + path, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def record(name, start, **data):
    row = dict(name=name, seconds=round(time.monotonic() - start, 3), **data)
    results.append(row)
    args.report.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(row), flush=True)


def body(messages, **options):
    return dict(model='qwen-flash', messages=messages, temperature=0, seed=91,
                max_tokens=48, reasoning_effort='none', **options)


def chat(name, payload, stream=False):
    expected = post('/v1/chat/completions/input_tokens', payload)['input_tokens']
    start = time.monotonic()
    if stream:
        payload = dict(payload, stream=True, stream_options={'include_usage': True})
        with requests.post(args.url + '/v1/chat/completions', json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            chunks = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith(b'data: ') and line != b'data: [DONE]']
        usages = [c['usage'] for c in chunks if c.get('usage')]
        assert usages, 'stream omitted requested final usage'
        usage = usages[-1]
        text = ''.join(c.get('delta', {}).get('content') or '' for chunk in chunks for c in chunk.get('choices', []))
        result = {'choices': [{'message': {'role': 'assistant', 'content': text}}]}
    else:
        result = post('/v1/chat/completions', payload)
        usage = result['usage']
    assert usage['prompt_tokens'] == expected, (name, expected, usage)
    cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
    assert 0 <= cached <= expected
    assert usage['total_tokens'] == expected + usage['completion_tokens'], usage
    record(name, start, usage=usage, expected_prompt_tokens=expected,
           content=result['choices'][0]['message'].get('content'), timings=result.get('timings'))
    return result, cached


def records(n):
    rng = random.Random(902)
    return '\n'.join('record ' + str(i) + ': ' + ''.join(rng.choices('abcdefghijklmnpqrstuvwxyz', k=32)) for i in range(n))


def raw(name, prompt, owner, **options):
    start = time.monotonic()
    payload = dict(prompt=prompt, n_predict=12, temperature=0, seed=91,
                   cache_prompt=True, llama_user_id=owner)
    payload.update(options)
    out = post('/completion', payload)
    record(name, start, timings=out['timings'], tokens_evaluated=out['tokens_evaluated'], content=out['content'])
    return out


if args.phase == 'prepare-restart':
    prompt = 'Read the registry.\n' + records(1400) + '\nReport only how many records there are.\nAnswer:'
    out = raw('durable-prime', prompt, session)
    args.state.write_text(json.dumps(dict(prompt=prompt, owner=session, n=out['tokens_evaluated'])))
elif args.phase == 'after-restart':
    state = json.loads(args.state.read_text())
    prompt = state['prompt'] + ' 1400\nRepeat the complete text of record 1399, including its number.\nAnswer:'
    out = raw('durable-restart-append', prompt, state['owner'], n_predict=64)
    assert out['timings']['cache_n'] >= state['n'], out['timings']
    expected = records(1400).splitlines()[-1]
    assert expected in out['content'], (expected, out['content'])
    cold = raw('durable-restart-cold-reference', prompt, state['owner'], n_predict=64, cache_prompt=False)
    assert cold['timings']['cache_n'] == 0, cold['timings']
    assert cold['content'] == out['content'], (out['content'], cold['content'])
    print('SSD restart output matches uncached generation and the source record.', flush=True)
elif args.phase == 'parity':
    messages = [
        {'role': 'system', 'content': 'Return only the requested answer, without explanation.'},
        {'role': 'user', 'content': records(128) + '\nRemember: the access code is ORCHID. Reply READY.'},
    ]
    previous, _ = chat('parity-prime', body(messages, llama_user_id=session))
    for i, question in enumerate(['Return the access code.', 'Write a Python function add(a, b) that returns their sum.', 'Return the access code followed by the number 128.']):
        messages += [previous['choices'][0]['message'], {'role': 'user', 'content': question}]
        payload = body(messages, llama_user_id=session)
        payload['max_tokens'] = 96
        warm, cached = chat('parity-warm-' + str(i), payload)
        assert cached > 2000, cached
        cold, _ = chat('parity-cold-' + str(i), dict(payload, cache_prompt=False))
        assert warm['choices'][0]['message']['content'] == cold['choices'][0]['message']['content'], (warm, cold)
        retry, cached_retry = chat('parity-exact-retry-' + str(i), payload)
        assert retry['choices'][0]['message']['content'] == cold['choices'][0]['message']['content']
        assert cached_retry < results[-1]['expected_prompt_tokens'], 'retry must decode suffix tokens for logits'
        previous = warm
    print('Warm/cold/exact-retry parity passed for three MTP turns.', flush=True)
else:
    # A new identity must be able to acquire an idle slot regardless of similarity.
    chat('explicit-owner-a', body([{'role': 'user', 'content': 'Reply with ALPHA.'}], llama_user_id=session + '-a'))
    chat('explicit-owner-b', body([{'role': 'user', 'content': 'Compute 9 minus 4; reply with the number.'}], llama_user_id=session + '-b'))
    messages = [
        {'role': 'system', 'content': 'You help maintain a synthetic registry. Answer briefly.'},
        {'role': 'user', 'content': 'Keep these records for later.\n' + records(320) + '\nReply with READY.'},
    ]
    first, _ = chat('auto-session-prime', body(messages))
    messages += [first['choices'][0]['message'], {'role': 'user', 'content': 'How many records did I supply? Reply with the number.'}]
    _, cached = chat('auto-session-stream-append', body(messages), stream=True)
    assert cached > 7000, cached
    # Pi compaction changes the first user/system content and hence automatic ID.
    summary_body = body([
        {'role': 'system', 'content': 'Summarize a coding conversation for another assistant. Include relevant facts and the next action. Be concise.'},
        {'role': 'user', 'content': '<conversation>\n' + json.dumps(messages) + '\n</conversation>\nSummarize the conversation in two sentences.'},
    ])
    summary_body['max_tokens'] = 128
    compacted, _ = chat('pi-shaped-compaction', summary_body, stream=True)
    chat('post-compaction-turn', body([
        {'role': 'system', 'content': 'Continue this coding conversation.'},
        {'role': 'user', 'content': 'Prior summary:\n' + compacted['choices'][0]['message']['content'] + '\nWhat should we do next? One sentence.'},
    ]))
    # Anthropic input + cached input must equal the independently counted prompt.
    anthropic = dict(model='qwen-flash', max_tokens=48, temperature=0,
                     messages=[{'role': 'user', 'content': 'Reply with the word READY.'}])
    for i in range(2):
        expected = post('/v1/messages/count_tokens', anthropic)['input_tokens']
        start = time.monotonic()
        out = post('/v1/messages', anthropic)
        usage = out['usage']
        assert usage['input_tokens'] + usage.get('cache_read_input_tokens', 0) + usage.get('cache_creation_input_tokens', 0) == expected, usage
        record('anthropic-usage-' + str(i), start, usage=usage, expected_prompt_tokens=expected)
        anthropic['messages'] += [{'role': 'assistant', 'content': out['content']}, {'role': 'user', 'content': 'Repeat that word.'}]
    # Invalid request must not retain a slot/count.
    start = time.monotonic()
    bad = requests.post(args.url + '/v1/chat/completions', json={'messages': 'invalid'}, timeout=10)
    assert bad.status_code == 400, bad.text
    record('bad-request', start, status=bad.status_code)
    chat('after-bad-request', body([{'role': 'user', 'content': 'Reply with OK.'}]))
    # Stop during prompt processing; immediate headers are sent before prefill.
    start = time.monotonic()
    payload = body([{'role': 'user', 'content': records(1400) + '\nSummarize these records.'}])
    payload.update(stream=True, stream_options={'include_usage': True})
    with requests.post(args.url + '/v1/chat/completions', json=payload, stream=True, timeout=120) as r:
        r.raise_for_status()
        time.sleep(0.2)
    record('cancel-prefill', start)
    chat('after-prefill-cancel', body([{'role': 'user', 'content': 'Reply with OK.'}]))
    # Disconnect after receiving a generated token, then reuse the same identity.
    start = time.monotonic()
    cancel_owner = session + '-cancel'
    payload = body([{'role': 'user', 'content': 'Write a detailed explanation of recursion, followed by several examples.'}], llama_user_id=cancel_owner)
    payload.update(stream=True, max_tokens=1024)
    with requests.post(args.url + '/v1/chat/completions', json=payload, stream=True, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines(chunk_size=1):
            if line.startswith(b'data: ') and line != b'data: [DONE]':
                chunk_data = json.loads(line[6:])
                if any(c.get('delta', {}).get('content') for c in chunk_data.get('choices', [])):
                    break
    record('cancel-generation', start)
    # Cancellation becomes visible after the current decode batch finishes.
    deadline = time.monotonic() + 20
    while any(s['is_processing'] for s in requests.get(args.url + '/slots', timeout=10).json()):
        assert time.monotonic() < deadline, 'cancelled generation retained its slot'
        time.sleep(0.05)
    chat('after-generation-cancel', body([{'role': 'user', 'content': 'Reply with OK.'}], llama_user_id=cancel_owner))
    # Force a tool invocation and complete the tool-result round trip.
    messages = [{'role': 'user', 'content': 'Use lookup_record to look up record 17.'}]
    tool_body = body(messages)
    tool_body.update(tools=[{'type': 'function', 'function': {'name': 'lookup_record', 'description': 'Look up a registry record.', 'parameters': {'type': 'object', 'properties': {'id': {'type': 'integer'}}, 'required': ['id']}}}], tool_choice='required')
    tool, _ = chat('tool-call', tool_body)
    msg = tool['choices'][0]['message']; call = msg['tool_calls'][0]
    assert call['function']['name'] == 'lookup_record' and json.loads(call['function']['arguments'])['id'] == 17
    tool_body.update(messages=messages + [msg, {'role': 'tool', 'tool_call_id': call['id'], 'content': 'Record 17 has status GREEN.'}], tool_choice='none')
    chat('tool-result', tool_body)
    # Synthetic solid-red PNG exercises actual media + subsequent text isolation.
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', 64, 64, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress((b'\0' + b'\xff\0\0' * 64) * 64)) + chunk(b'IEND', b'')
    image_url = 'data:image/png;base64,' + base64.b64encode(png).decode()
    vision, _ = chat('vision', body([{'role': 'user', 'content': [{'type': 'text', 'text': 'What color fills this image? One word.'}, {'type': 'image_url', 'image_url': {'url': image_url}}]}]))
    assert 'red' in vision['choices'][0]['message']['content'].lower()
    chat('text-after-vision', body([{'role': 'user', 'content': 'Reply with OK.'}]))
    slots = requests.get(args.url + '/slots', timeout=10).json()
    assert all(not s['is_processing'] for s in slots), slots
    print('All live exercise checks passed.', flush=True)
