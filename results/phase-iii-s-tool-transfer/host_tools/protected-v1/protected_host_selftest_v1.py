"""Synthetic host-access fixtures only. No model or protected roster is read."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch


def run(out):
    wall,cpu=time.perf_counter(),time.process_time()
    source=Path(__file__).with_name('protected_host_v1.py')
    spec=importlib.util.spec_from_file_location('private_protected_host',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tests={}
    def refused(name,fn,expected):
        try:fn()
        except ValueError as exc:tests[name]=expected in str(exc)
        else:tests[name]=False
    request={'study':str(module.PRIVATE/'study-v1'),'arm':'sequential','ledger':str(module.PRIVATE/'execution'),
        'original':'original','decimal_parent':'r8'}
    # There is no chance to read a source, measurement or roster after this gate.
    with patch.object(module.host,'_initialized',return_value=((module.PRIVATE/'study-v1').resolve(),{},{})),\
         patch.object(module,'file_hash',return_value=module.BRIDGE_SHA),\
         patch.object(module.study,'validate_arm_locks',side_effect=ValueError('both decisions missing')),\
         patch.object(module,'read',side_effect=AssertionError('No bank or candidate metadata read is allowed yet')):
        refused('both_arm_decisions_required_before_access',lambda:module.context(request,required_stage='qualification'),'both decisions missing')
    with tempfile.TemporaryDirectory(prefix='wb-s-host-fixture-') as temp:
        temp=Path(temp)
        ctx={'protocol':{'arm':'sequential','source_sha256':{'file':'a'}},'local':temp,'root':temp,
            'candidate':temp/'merged','reconstructed':temp/'rebuilt','actual_parent':temp/'parent',
            'lock':{'representation_sha256':'c'*64}}
        for which,gate,replay in (('failed_behavior',False,True),('failed_replay',True,False)):
            with patch.object(module,'context',return_value=ctx),\
                 patch.object(module,'read',return_value={'passed':True,'exact_replays_passed':True,
                    'driver_sha256':'hash','gate_sha256':'hash'}),\
                 patch.object(module,'file_hash',return_value='hash'),\
                 patch.object(module,'validate_formal_replays',return_value=({'passed':gate},{'passed':replay})),\
                 patch.object(module,'worker',side_effect=AssertionError('A later bank must not run')):
                refused('confirmation_refuses_'+which,lambda:module.run({'action':'confirmation'},temp/'request.json'),
                    'Confirmation refused')
                tests['no_directory_created_'+which]=not (temp/'protected-v1').exists()
        protocol={'source_sha256':{'file':'a'},'task_manifest_sha256':'m','endpoints':{},'tool_completion':{}}
        fake={'path':'worker','rows':'rows','authorization':'auth','protocol':'protocol',
            'stage':'qualification','execution':'execution'}
        with patch.object(module,'read',return_value={**protocol,'task_manifest_sha256':'different'}),\
             patch.object(module,'validate_worker',side_effect=AssertionError('Foreign cached row must not be scored')):
            refused('foreign_parent_cache_refused_before_scoring',
                lambda:module.check_worker({'protocol':protocol},fake,'sha',temp,'tool','qualification'),
                'cached parent changed')
        with patch.object(module,'read',return_value={'command':['python','-m','worker','--base',str(temp/'merged')]}),\
             patch.object(module,'file_hash',return_value='hash'):
            refused('replay_from_wrong_physical_directory_refused',lambda:module.check_base(fake,temp/'rebuilt'),
                'physical checkpoint')
        # A top-level replay PASS cannot conceal an actual native mismatch.
        references={'candidate':{},'r8':{}}
        pairs={}
        for role in ('candidate','parent'):
            for task in ('tool','decimal'):
                key=f'qualification/{role}/{task}'
                original={**fake,'path':str(temp/f'{role}-{task}.json'),'execution':str(temp/f'{role}-{task}-execution.json')}
                replayed={**fake,'path':str(temp/f'{role}-{task}-replay.json'),'execution':str(temp/f'{role}-{task}-replay-execution.json')}
                references['candidate' if role=='candidate' else 'r8'][task]=original
                pairs[key]={'recorded':original,'replayed':replayed}
        saved={'passed':True,'replays':pairs,'exact_native':{k:True for k in pairs},
            'source_sha256':ctx['protocol']['source_sha256'],'gate_sha256':'hash'}
        ctx['protocol']['checkpoint_sha256']='p'*64
        with patch.object(module.study,'replay_stage',return_value={'workers':references}),\
             patch.object(module,'read',return_value=saved),patch.object(module,'file_hash',return_value='hash'),\
             patch.object(module,'check_worker',return_value=({'measurement':{}},{'passed':True})),\
             patch.object(module,'exact_native_replay',return_value=False):
            refused('rewritten_replay_pass_refused',lambda:module.validate_formal_replays(ctx,'qualification'),
                'does not reproduce')
    result={'schema':'weightbridge.phase_iiis.private_protected_host_selftest.v1',
        'passed':all(tests.values()),'tests':tests,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'selftest_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model_calls':0,'protected_roster_reads':0,'training_or_scientific_source_changes':False,
        'resources':{'wall_seconds':time.perf_counter()-wall,'cpu_seconds':time.process_time()-cpu,'peak_memory_bytes':None},
        'scope':'Mocked negative authorization/cache/replay fixtures. No positive model or protected-stage execution is established.'}
    with Path(out).open('x',encoding='utf8',newline='\n') as stream:
        json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    if not result['passed']:raise AssertionError(result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    run(parser.parse_args().out)
