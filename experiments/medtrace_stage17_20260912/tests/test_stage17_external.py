import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.medtrace.stage17_external import receive, validate_phase, worker


class ExternalTests(unittest.TestCase):
    def test_mixed_precision_import_requires_each_original_phase_binding(self):
        import copy
        from scripts.medtrace.stage17_external import phase_config
        base=dict(freeze_id='f',N=1,order=['e'],runtime_lock={'dtype':'float16'},
            code_commit='original',methods={'generation':{'dtype':'float16'}})
        seq=copy.deepcopy(base);seq.update(code_commit='bf16',precision_variant='LORA_SEQUENTIAL_BF16_STABILITY_V1')
        seq['runtime_lock']['dtype']='bfloat16';seq['methods']['generation']['dtype']='bfloat16'
        cfg=dict(base,acceptance_amendment={'id':'LORA_SINGLE_FP16_SEQUENTIAL_BF16_V1'},
            phase_assignments={'single':base,'sequential':seq})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for mode,source in [('single',base),('sequential',seq)]:
                p=root/'private'/f'lora_{mode}';p.mkdir(parents=True)
                binding=dict(freeze_id='f',method='lora',mode=mode,runtime=source['runtime_lock'],
                    code_commit=source['code_commit'],method_lock=source['methods'],order=['e'],prefixes=[1])
                for name,value in [('BINDING',binding),('COMPLETE',dict(status='GENERATED_NOT_SCORED',N=1,phase=binding)),('CLEANUP',dict(status='DELETED'))]:
                    (p/f'{name}.json').write_text(json.dumps(value))
                validate_phase(root,cfg,mode)
            broken=copy.deepcopy(cfg);broken['phase_assignments']['sequential']['order']=['other']
            with self.assertRaises(ValueError): phase_config(broken,'lora','sequential')
            with self.assertRaises(ValueError): validate_phase(root,base,'sequential')
            with self.assertRaises(ValueError): worker(dict(cfg,run=d))

    def test_assigned_baselines_keep_method_dependencies_and_import_separately(self):
        from scripts.medtrace.stage17_external import assigned_phases
        from scripts.medtrace.stage17_campaign import prefixes
        cfg=dict(assigned_methods=['grace','belora'],freeze_id='f',runtime_lock={},
            code_commit='c',methods={},order=['e'],N=1)
        phases=assigned_phases(cfg)
        self.assertEqual(phases,[('grace','single'),('grace','sequential'),('belora','single'),('belora','sequential')])
        with self.assertRaises(ValueError): assigned_phases(dict(cfg,assigned_methods=['grace','lora']))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'private').mkdir()
            (root/'private/EXTERNAL_BASELINES.json').write_text(json.dumps(cfg))
            for method,mode in phases:
                p=root/'private/external-incoming-baselines/private'/f'{method}_{mode}';p.mkdir(parents=True)
                binding=dict(freeze_id='f',method=method,mode=mode,runtime={},code_commit='c',
                    method_lock={},order=['e'],prefixes=prefixes(1))
                for name,value in [('BINDING',binding),('COMPLETE',dict(status='GENERATED_NOT_SCORED',N=1,phase=binding)),('CLEANUP',dict(status='DELETED'))]:
                    (p/f'{name}.json').write_text(json.dumps(value))
            receive(root,'EXTERNAL_BASELINES')
            self.assertTrue((root/'private/EXTERNAL_BASELINES_IMPORTED.json').exists())
            self.assertFalse((root/'private/EXTERNAL_LORA_IMPORTED.json').exists())
            for method,mode in phases: validate_phase(root,cfg,mode,method=method)

    def test_authorized_variant_runs_sequential_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'private').mkdir(); (root/'public').mkdir()
            (root/'private/COHORT_AND_SUPPORT_LEDGER.json').write_text('{}')
            cfg=dict(run=d,source_run=d,gpu=2,N=146,python='python',entry='run.py',
                precision_variant='LORA_SEQUENTIAL_BF16_STABILITY_V1',assigned_modes=['sequential'])
            with patch('scripts.medtrace.stage17_campaign.role_map',return_value={}), \
                 patch('scripts.medtrace.stage17_campaign.cleanup') as cleanup, \
                 patch('scripts.medtrace.stage17_external.validate_phase'), \
                 patch('scripts.medtrace.stage17_external.subprocess.check_output',return_value='40000'), \
                 patch('scripts.medtrace.stage17_external.subprocess.run') as launched:
                worker(cfg)
            self.assertEqual(launched.call_count,1)
            self.assertEqual(launched.call_args.kwargs['env']['JOB_MODE'],'sequential')
            cleanup.assert_called_once_with(cfg,'lora','sequential')

    def test_completed_single_is_not_retrained_when_cleanup_visibility_is_blocked(self):
        from scripts.medtrace.stage17_campaign import CheckpointVisibilityError
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'private/lora_single').mkdir(parents=True); (root/'public').mkdir()
            (root/'private/lora_single/COMPLETE.json').write_text('{}')
            (root/'private/PREFIX_ACTIVE_TARGETS.json').write_text('{}')
            (root/'private/COHORT_AND_SUPPORT_LEDGER.json').write_text('{}')
            cfg=dict(run=d,source_run=d,gpu=0,N=146,python='python',entry='run.py')
            with patch('scripts.medtrace.stage17_campaign.role_map',return_value={}), \
                 patch('scripts.medtrace.stage17_campaign.cleanup',side_effect=[CheckpointVisibilityError('denied'),None]), \
                 patch('scripts.medtrace.stage17_external.validate_phase'), \
                 patch('scripts.medtrace.stage17_external.subprocess.check_output',return_value='24000'), \
                 patch('scripts.medtrace.stage17_external.subprocess.run') as launched:
                worker(cfg)
            self.assertEqual(launched.call_count,1)
            self.assertEqual(launched.call_args.kwargs['env']['JOB_MODE'],'sequential')
            status=json.loads((root/'public/PROGRESS.json').read_text())
            self.assertEqual(status['status'],'GPU_GENERATED_CLEANUP_PENDING')
            self.assertEqual(status['cleanup_pending'],['single'])
            self.assertFalse((root/'private/lora_single/CLEANUP.json').exists())

    def test_only_complete_bound_cleaned_phases_are_adopted_without_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'private').mkdir()
            cfg=dict(freeze_id='f',runtime_lock={},code_commit='c',methods={},order=['e'],N=1)
            def put(p,value):
                p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(value))
            put(root/'private/EXTERNAL_LORA.json',cfg)
            incoming=root/'private/external-incoming'
            for mode in ('single','sequential'):
                directory=incoming/'private'/f'lora_{mode}'
                phase=dict(freeze_id='f',method='lora',mode=mode,runtime={},code_commit='c',
                    method_lock={},order=['e'],prefixes=[1])
                put(directory/'BINDING.json',phase)
                put(directory/'COMPLETE.json',dict(status='GENERATED_NOT_SCORED',N=1,phase=phase))
                put(directory/'CLEANUP.json',dict(status='DELETED'))
            with self.assertRaises(ValueError):
                validate_phase(incoming,dict(cfg,code_commit='wrong'),'single')
            collision=root/'private/lora_single'; collision.mkdir()
            with self.assertRaises(RuntimeError): receive(root)
            self.assertFalse((root/'private/EXTERNAL_LORA_IMPORTED.json').exists())
            collision.rmdir(); receive(root)
            self.assertEqual(json.loads((root/'private/EXTERNAL_LORA_IMPORTED.json').read_text()),cfg)
            validate_phase(root,cfg,'single'); validate_phase(root,cfg,'sequential')


if __name__=='__main__': unittest.main()
