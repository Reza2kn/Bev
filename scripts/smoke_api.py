#!/usr/bin/env python3
"""Small functional and latency checks; run on Stallion, not the Mac.

These hand-authored checks verify API behavior. They are not benchmark accuracy.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:18781')
    parser.add_argument('--out', type=Path, default=Path('artifacts/api-smoke.json'))
    args = parser.parse_args()
    results = []
    with httpx.Client(base_url=args.url, timeout=300, trust_env=False) as client:
        health = client.get('/health').raise_for_status().json()
        def run(name, payload, expected):
            before = time.perf_counter()
            response = client.post('/v1/decisions', json=payload)
            response.raise_for_status()
            answer = response.json()
            valid = all(
                len(field['candidates']) == (2 if payload['schema'][key]['type'] == 'boolean' else len(payload['schema'][key]['choices']))
                and abs(sum(c['probability'] for c in field['candidates']) - 1) < 1e-6
                and field['backend_requests'] == 1
                for key, field in answer['fields'].items()
            )
            result = {'name': name, 'expected': expected, 'actual': answer['parsed_json'],
                      'passed': answer['parsed_json'] == expected and valid,
                      'http_ms': (time.perf_counter()-before)*1000,
                      'server_ms': answer['elapsed_ms'], 'response': answer}
            results.append(result)
            return answer
        run('negation', {'context': 'The order has shipped. It has not been delivered.', 'schema': {
            'delivered': {'type': 'boolean', 'description': 'Has the order been delivered?'}}}, {'delivered': False})
        run('missing_evidence', {'context': 'Alex ordered a sandwich.', 'schema': {
            'payment': {'type': 'enum', 'description': 'Which payment method did Alex use?',
                        'choices': ['cash', 'card', 'unknown']}}}, {'payment': 'unknown'})
        run('unicode', {'context': 'مشتری می‌گوید مبلغ دو بار از حسابش کم شده است.', 'schema': {
            'route': {'type': 'enum', 'description': 'Choose the support queue: a duplicate charge is billing.',
                      'choices': ['صورتحساب', 'فنی', 'نامشخص']}}}, {'route': 'صورتحساب'})
        options = [f'ID_{i:03d}' for i in range(255)]
        run('all_255_choices', {'context': 'The exact selected record identifier is ID_203.', 'schema': {
            'record': {'type': 'enum', 'description': 'Choose the exact record identifier explicitly stated in the context.',
                       'choices': options}}}, {'record': 'ID_203'})
        body = {'context': 'A tight bend is ahead and grip is low. Boost is charged.', 'schema': {
            'maneuver': {'type': 'enum', 'description': 'Brake for a tight bend with low grip.', 'choices': ['brake', 'coast', 'accelerate']},
            'boost': {'type': 'boolean', 'description': 'Activate boost only on a clear straight with good grip.'}}}
        # First call and repeats are separately labeled: neither is guaranteed a
        # process-cold run because the runtime may already cache other requests.
        repeats = [run(f'repeat_{i}', body, {'maneuver': 'brake', 'boost': False}) for i in range(6)]
        for name, change in [('unsupported_alignment', {'strategy': 'aligned_prefill'}), ('unsupported_cache_salt', {'cache_salt': 'test'})]:
            response = client.post('/v1/decisions', json={**body, **change})
            results.append({'name': name, 'passed': response.status_code == 422, 'status': response.status_code})
        response = client.post('/v1/decisions', json={'context': 'long '+('x '*18000), 'schema': {'x': {'type': 'boolean', 'description': 'Is this short?'}}})
        results.append({'name':'reject_oversize', 'passed':response.status_code==422 and 'context window' in response.text, 'status':response.status_code, 'detail': response.json()})
        req = {'model': 'bev-bonsai-27b', 'state': 'The correct code is second.', 'questions': {
            'q': {'type': 'choice', 'instructions': 'Choose the code explicitly named.', 'criteria': {'first': 'first', 'second': 'second'}}}}
        response = client.post('/v1/systemone', json=req).raise_for_status().json()
        results.append({'name':'systemone', 'passed':response['answers']['q']['choice']=='second' and set(response['answers']['q']['probabilities'])=={'first','second'},'response':response})
    report = {'kind': 'hand_authored_functional_checks_not_benchmark', 'health': health,
              'passed': sum(r['passed'] for r in results), 'total': len(results), 'results': results,
              'repeated_two_field_server_median_ms': statistics.median(x['elapsed_ms'] for x in repeats[1:]),
              'first_two_field_server_ms': repeats[0]['elapsed_ms'],
              'latency_scope': 'API-side elapsed including prompt rendering, tokenization, queueing and two independent field scoring requests; first/repeated, not cold GPU or throughput benchmark'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','health')}, indent=2))
    raise SystemExit(0 if report['passed']==report['total'] else 1)


if __name__ == '__main__':
    main()
