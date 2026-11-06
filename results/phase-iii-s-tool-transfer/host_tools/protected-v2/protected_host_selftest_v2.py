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
    source=Path(__file__).with_name('protected_host_v2.py')
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
                 patch.object(module,'formal_closure',return_value={'completed_valid':True,'passed':gate and replay}),\
                 patch.object(module,'worker',side_effect=AssertionError('A later bank must not run')):
                refused('confirmation_refuses_'+which,lambda:module.run({'action':'confirmation'},temp/'request.json'),
                    'Confirmation refused')
                tests['no_directory_created_'+which]=not (temp/'protected-v2').exists()
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
        def closure_fixture(name,*,marker=True,failed=False,behavior=True,replay=True,disposition='BEHAVIOR_QUALIFIED_PENDING_REPLAY'):
            root=temp/name;local=root/'sequential';directory=local/'protected-v2/qualification'
            directory.mkdir(parents=True)
            pp=local/'protocol.json';lockpath=root/'locks/sequential.json';lockpath.parent.mkdir()
            both=root/'locks/both-arms.json';request_path=root/'request.json'
            protocol={'arm':'sequential','task_manifest_sha256':'m'*64,'source_sha256':{},'checkpoint_sha256':'p'*64}
            lock={'representation_sha256':'c'*64}
            for path,value in ((pp,protocol),(lockpath,lock),(both,{}),(directory/'gate.json',{}),
                    (directory/'replays.json',{}),(directory/'started.json',{})):
                path.write_text(json.dumps(value),encoding='utf8')
            request={'schema':'weightbridge.phase_iiis.protected_host_request.v2','driver_sha256':module.file_hash(source),
                'action':'qualification','arm':'sequential'}
            request_path.write_text(json.dumps(request),encoding='utf8')
            context={'local':local,'root':root,'protocol':protocol,'protocol_path':pp,'lock':lock,'lock_path':lockpath,
                'original':root/'original','actual_parent':root/'parent','reconstruction_gate':root/'reconstruction.json'}
            completed={'schema':'weightbridge.phase_iiis.private_protected_execution.v2','action':'qualification','arm':'sequential',
                'driver_sha256':module.file_hash(source),'bridge_sha256':module.BRIDGE_SHA,'protocol_sha256':module.file_hash(pp),
                'task_manifest_sha256':protocol['task_manifest_sha256'],'representation_sha256':lock['representation_sha256'],
                'lock_sha256':module.file_hash(lockpath),'all_arms_lock_sha256':module.file_hash(both),
                'completed':True,'passed':behavior and replay,'exact_replays_passed':replay,'disposition':disposition if replay else 'NATIVE_REPLAY_FAILURE',
                'gate_sha256':module.file_hash(directory/'gate.json'),'replay_receipt_sha256':module.file_hash(directory/'replays.json'),
                'request_sha256':module.file_hash(request_path)}
            completed['input_sha256']=module.commitment({'protocol':pp,'lock':lockpath,'all_arms_lock':both,
                'gate':directory/'gate.json','replays':directory/'replays.json','host_source':source,'host_request':request_path,
                'started':directory/'started.json'})
            if marker:(directory/'completed.json').write_text(json.dumps(completed),encoding='utf8')
            if failed:(directory/'failed.json').write_text(json.dumps({'error':'fixture interruption'}),encoding='utf8')
            measured=({'passed':behavior,'disposition':disposition},{'passed':replay,'replays':{}})
            return context,directory,measured,request_path

        interrupted,idir,_,_=closure_fixture('interrupted',marker=False)
        with patch.object(module,'validate_formal_replays',side_effect=AssertionError('Partial replay files must not be promoted')):
            c=module.formal_closure(interrupted,'qualification')
            tests['gate_and_replay_without_completion_are_interrupted']=(c['disposition']=='INTERRUPTED_INCOMPLETE'
                and not c['completed_valid'] and c['gate_path'] is None and c['replays']=={})
        failed,fdir,_,_=closure_fixture('failed-despite-completed',failed=True)
        with patch.object(module,'validate_formal_replays',side_effect=AssertionError('Failed host stage must not be promoted')):
            c=module.formal_closure(failed,'qualification')
            tests['failed_record_blocks_even_existing_completed_marker']=c['disposition']=='INTERRUPTED_INCOMPLETE' and not c['passed']
        good,gdir,measured,_=closure_fixture('valid-completed')
        with patch.object(module,'validate_formal_replays',return_value=measured):
            c=module.formal_closure(good,'qualification')
            tests['fully_bound_completed_fixture_is_eligible']=c['completed_valid'] and c['passed']
        insuff,_,measured,_=closure_fixture('insufficient',behavior=False,disposition='INSUFFICIENT_IMPROVEMENT_HEADROOM')
        with patch.object(module,'validate_formal_replays',return_value=measured):
            c=module.formal_closure(insuff,'qualification')
            tests['completed_insufficient_headroom_remains_failed']=c['completed_valid'] and not c['passed'] and c['disposition']=='INSUFFICIENT_IMPROVEMENT_HEADROOM'
        mismatch,mdir,measured,_=closure_fixture('marker-tamper')
        wrong=json.loads((mdir/'completed.json').read_bytes());wrong['lock_sha256']='wrong'
        (mdir/'completed.json').write_text(json.dumps(wrong),encoding='utf8')
        with patch.object(module,'validate_formal_replays',side_effect=AssertionError('Changed marker must fail before replay')):
            refused('completed_wrong_lock_identity_refused',lambda:module.formal_closure(mismatch,'qualification'),'exact driver, protocol, lock')
        unopened={**good,'local':temp/'unopened/sequential'}
        with patch.object(module,'read',side_effect=AssertionError('Unopened bank must not be read')):
            c=module.formal_closure(unopened,'confirmation')
            tests['unopened_stage_is_null_without_read']=c['disposition']=='UNOPENED' and c['gate_path'] is None and not c['passed']
        # End-to-end host finalization: even convincing partial gate/replay files
        # cannot be forwarded to the frozen verifier without completed markers.
        final_context,_,_,_=closure_fixture('interrupted-finalization',marker=False)
        final_request=temp/'interrupted-finalization/final-request.json'
        final_request.write_text(json.dumps({'action':'finalize'}),encoding='utf8')
        called={}
        def frozen_final(out,*args,**kwargs):
            called.update(kwargs)
            value={'passed':False,'checks':{'all_native_replays':False},'disposition':'DEVELOPMENT_GAIN_WITHOUT_SEALED_CONFIRMATION'}
            module.write_once(out,value);return value
        with patch.object(module,'context',return_value=final_context),patch.object(module.study,'finalize',side_effect=frozen_final),\
             patch.object(module,'worker',side_effect=AssertionError('Finalization is evidence only')):
            result=module.run({'action':'finalize'},final_request)
            tests['interrupted_finalization_never_forwards_partial_pass']=(not result['passed']
                and result['disposition']=='INTERRUPTED_INCOMPLETE' and called['qualification_path'] is None
                and called['confirmation_path'] is None and called['replays']=={})
    result={'schema':'weightbridge.phase_iiis.private_protected_host_selftest.v2',
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
