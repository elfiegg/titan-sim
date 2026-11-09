import json
from pathlib import Path
import tempfile
import unittest
from titan_sim.runtime import check_runtime


class RuntimeHandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = dict(schema_version=1, run_id='case', status='validated',
            final_image={'sha256':'a'*64}, codebase={'resolved_commit':'b'*40},
            recipe={'module':'recipes', 'name':'debug', 'overrides':{}},
            target={'gpu_model':'GB200'}, reproduce={'command':'python train.py'},
            validation={'capability_report':'capability-report.json'})
        self.check = dict(id='recipe-training', required=True, status='pass', exit_code=0,
            command='python train.py', assertions=['finite loss'], evidence_paths=['training.log'],
            image_identity='sha256:'+'a'*64, target={'gpu_model':'GB200'})
        self.report = {'schema_version':1, 'run_id':'case'}
        (self.root/'training.log').write_text('finite loss observed\n')

    def run_check(self, **kwargs):
        (self.root/'runtime-manifest.json').write_text(json.dumps(self.manifest))
        (self.root/'capability-report.json').write_text(json.dumps(dict(self.report, checks=[self.check])))
        return check_runtime(self.root/'runtime-manifest.json', **kwargs)

    def test_ready_pins_artifacts_without_importing_framework(self):
        result = self.run_check(expected={'codebase_commit':'b'*40,'required_checks':['recipe-training']})
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(len(result['runtime_manifest_sha256']),64)

    def test_changed_recipe_and_missing_feature_are_blocked(self):
        result = self.run_check(expected={'recipe':{'name':'full'},'required_checks':['hybrid-ep']})
        self.assertEqual(result['status'],'blocked')
        self.assertTrue(any('recipe/overrides' in x for x in result['unresolved']))
        self.assertTrue(any('hybrid-ep' in x for x in result['unresolved']))

    def test_claimed_pass_requires_evidence_and_exact_image(self):
        (self.root/'training.log').unlink()
        self.check['image_identity']='sha256:'+'c'*64
        result=self.run_check()
        self.assertEqual(result['status'],'blocked')
        self.assertTrue(any('evidence is missing' in x for x in result['unresolved']))
        self.assertTrue(any('image identity differs' in x for x in result['unresolved']))

    def test_untested_and_partial_never_become_ready(self):
        self.manifest['status']='partial'
        self.check['status']='untested'
        self.assertEqual(self.run_check()['status'],'blocked')

    def test_unknown_remote_evidence_requires_staging(self):
        self.check['evidence_paths']=['https://example.invalid/log']
        self.assertEqual(self.run_check()['status'],'blocked')

    def test_inconsistent_report_identity_and_schema_block(self):
        for field,value in [('schema_version',999),('run_id','another-run'),('recipe',{'name':'different'})]:
            with self.subTest(field=field):
                previous=self.report.copy()
                self.report[field]=value
                self.assertEqual(self.run_check()['status'],'blocked')
                self.report=previous

    def test_manifest_required_check_cannot_be_silently_omitted(self):
        self.manifest['validation']['required_checks']=['missing-probe']
        self.assertEqual(self.run_check()['status'],'blocked')

    def test_evidence_content_is_fingerprinted(self):
        before=self.run_check()['evidence_sha256']
        (self.root/'training.log').write_text('changed evidence')
        self.assertNotEqual(before,self.run_check()['evidence_sha256'])
