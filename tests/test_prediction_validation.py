import copy
import hashlib
import json
import sys
import types
import unittest

from titan_sim.core import Node
from titan_sim.prediction_validation import freeze_prediction, compare_control, run_control
from titan_sim.workload import Workload


def fixture():
    workload = Workload([Node(0,'native',0,attrs={'cost_key':'case'})],[],
        dict(ranks=[0],world_size=1,rank_placement={'0':'gpu0'},prediction_scope='training_update',
             useful_tokens=100,unresolved=[]),dict(id='fixture',capacity_bytes=1000))
    samples=[0.5,0.5]
    costs={'case':dict(seconds=0.5,samples_seconds=samples,hardware_id='fixture',
        evidence_sha256=hashlib.sha256(json.dumps(samples,separators=(',',':')).encode()).hexdigest())}
    context=dict(recipe={'fixture':'unchanged'},runtime={'fixture':'native'},hardware=workload.hardware,
        placement=workload.manifest['rank_placement'],scope='training_update',useful_tokens=100,
        timing_metric='completed_scope_device_seconds')
    workload.manifest['validation_context']=copy.deepcopy(context)
    return workload,costs,context


class PredictionValidationTests(unittest.TestCase):
    def control(self, context):
        return dict(context=copy.deepcopy(context),measurement_kind='fresh_native_control',
            run_id='separate-control',evidence={'raw-observations':'a'*64},samples_seconds=[1.,1.2,.8])

    def test_complete_timing_comparison_does_not_require_memory_fit(self):
        workload,costs,context=fixture()
        frozen=freeze_prediction(workload,costs,context=context)
        report=compare_control(frozen,self.control(context))
        self.assertEqual(report['timing']['relative_error'],-0.5)
        self.assertEqual(report['timing']['observed_tokens_per_second'],100)
        self.assertEqual(frozen['prediction']['memory_verdict'],'inconclusive')

    def test_partial_rank_or_missing_semantics_cannot_be_full_update(self):
        for mutation in (lambda w:w.manifest.update(world_size=2),
                         lambda w:w.manifest.update(unresolved=['unknown overlap']),
                         lambda w:w.manifest.pop('prediction_scope')):
            workload,costs,context=fixture()
            mutation(workload)
            with self.assertRaises(ValueError):freeze_prediction(workload,costs,context=context)

    def test_failed_or_partial_capture_cannot_freeze_full_update(self):
        for status in ('blocked','incomplete'):
            for nested in (True,False):
                workload,costs,context=fixture()
                if nested:workload.manifest['source_manifests']=[{'capture_status':status}]
                else:workload.manifest['capture_status']=status
                with self.assertRaisesRegex(ValueError,'complete capture'):
                    freeze_prediction(workload,costs,context=context)

    def test_stale_prediction_and_calibration_reuse_rejected(self):
        workload,costs,context=fixture()
        frozen=freeze_prediction(workload,costs,context=context)
        control=self.control(context)
        control['evidence']={'raw':frozen['catalog_sha256']}
        with self.assertRaisesRegex(ValueError,'independent'):compare_control(frozen,control)
        frozen['prediction']['seconds']=4.
        with self.assertRaisesRegex(ValueError,'checksum'):compare_control(frozen,self.control(context))

    def test_recipe_runtime_must_match_recorded_source_identity(self):
        for key in ('recipe','runtime'):
            workload,costs,context=fixture()
            context[key]={'changed':True}
            with self.assertRaisesRegex(ValueError,'source identity'):
                freeze_prediction(workload,costs,context=context)
        workload,costs,context=fixture()
        del workload.manifest['validation_context']
        with self.assertRaisesRegex(ValueError,'source identity'):
            freeze_prediction(workload,costs,context=context)

    def test_changed_scope_tokens_and_runtime_rejected(self):
        workload,costs,context=fixture()
        frozen=freeze_prediction(workload,costs,context=context)
        for key,value in [('scope','native_window'),('useful_tokens',200),('runtime',{'different':True}),
                          ('timing_metric','completed_scope_wall_seconds')]:
            control=self.control(context)
            control['context'][key]=value
            with self.assertRaisesRegex(ValueError,'mismatch'):compare_control(frozen,control)

    def test_control_runner_does_not_receive_prediction_or_costs(self):
        workload,costs,context=fixture()
        frozen=freeze_prediction(workload,costs,context=context)
        module=types.ModuleType('prediction_control_fixture')
        def runner(received,config):
            self.assertEqual(received,context)
            self.assertNotIn('prediction',received)
            return self.control(received)
        module.runner=runner
        sys.modules[module.__name__]=module
        try:
            result=run_control(frozen,runner=module.__name__+':runner',config={})
            self.assertIn('before invoking',result['execution']['order'])
        finally:del sys.modules[module.__name__]

    def test_allocator_observations_are_not_physical_comparisons(self):
        workload,costs,context=fixture()
        frozen=freeze_prediction(workload,costs,context=context)
        control=self.control(context)
        control['memory']={'gpu0':{'metric':'process_reserved_bytes','peak_bytes':100}}
        result=compare_control(frozen,control)
        self.assertEqual(result['memory'][0]['comparison'],'incomparable')

    def test_invalid_samples_rejected(self):
        workload,costs,context=fixture()
        frozen=freeze_prediction(workload,costs,context=context)
        for samples in ([],[True],[float('nan')],[-1],[0]):
            control=self.control(context)
            control['samples_seconds']=samples
            with self.assertRaises(ValueError):compare_control(frozen,control)
