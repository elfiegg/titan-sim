import copy
import types
import unittest
from unittest.mock import patch

from titan_sim.plan_capture import export_recipe, export_trainer_plan


class PlanCaptureTests(unittest.TestCase):
    def trainer(self, fail=False):
        class Runtime:
            def __init__(self):
                self.prepared = False
            def ensure_ready(self, context):
                self.prepared = True
            def step(self):
                self.ensure_ready(None)
                # A native runtime may invoke readiness more than once per step.
                self.ensure_ready(None)

        class Trainer:
            def __init__(self):
                self.engine = types.SimpleNamespace(pp_schedule=Runtime())
                self.parameter = 4
                self.completed = 0
            def train_step(self):
                self.engine.pp_schedule.step()
                if fail:
                    raise RuntimeError('native failure')
                self.parameter -= self.parameter / 4
                self.completed += 1
            def should_continue_training(self):
                return self.completed < 5
            def train(self):
                while self.should_continue_training():
                    self.train_step()
        return Trainer()

    def extract(self, runtime, **kwargs):
        return {'schema_version': 'titan_sim.native_plan.v1', 'prepared': runtime.prepared,
                'schedule': {}, 'graphs': {}, 'stages': {},
                'unresolved': [], 'provenance': kwargs['provenance']}

    def test_prepared_export_does_not_run_trainer(self):
        trainer = self.trainer()
        with patch('titan_sim.native_plan.extract_plan', side_effect=self.extract):
            result = export_trainer_plan(trainer)
        self.assertFalse(result['prepared'])
        self.assertEqual(trainer.completed, 0)
        self.assertEqual(result['observation']['mode'], 'prepared_objects')

    def test_lazy_export_preserves_steps_and_restores_methods(self):
        trainer = self.trainer()
        reference = copy.deepcopy(trainer)
        original = trainer.train_step
        original_ready = trainer.engine.pp_schedule.ensure_ready
        with patch('titan_sim.native_plan.extract_plan', side_effect=self.extract):
            result = export_trainer_plan(trainer, observe_steps=2,
                                         provenance={'recipe': {'checkpoint': True}})
        reference.train_step()
        reference.train_step()
        self.assertEqual(trainer.parameter, reference.parameter)
        self.assertEqual(trainer.completed, 2)
        self.assertEqual(trainer.train_step, original)
        self.assertEqual(trainer.engine.pp_schedule.ensure_ready, original_ready)
        self.assertTrue(result['prepared'])
        self.assertEqual(result['provenance']['recipe'], {'checkpoint': True})

    def test_failure_is_not_replaced_by_partial_success(self):
        trainer = self.trainer(fail=True)
        original = trainer.train_step
        original_ready = trainer.engine.pp_schedule.ensure_ready
        with patch('titan_sim.native_plan.extract_plan', side_effect=self.extract):
            with self.assertRaisesRegex(RuntimeError, 'native failure'):
                export_trainer_plan(trainer, observe_steps=1)
        self.assertEqual(trainer.train_step, original)
        self.assertEqual(trainer.engine.pp_schedule.ensure_ready, original_ready)

    def test_dynamic_plan_changes_are_rejected(self):
        trainer = self.trainer()
        counter = 0
        def changing(*args, **kwargs):
            nonlocal counter
            counter += 1
            return {'variant': counter}
        with patch('titan_sim.native_plan.extract_plan', side_effect=changing):
            with self.assertRaisesRegex(ValueError, 'plan changed'):
                export_trainer_plan(trainer, observe_steps=2)

    def test_observer_failure_finishes_native_steps_before_raising(self):
        trainer = self.trainer()
        reference = copy.deepcopy(trainer)
        original_ready = trainer.engine.pp_schedule.ensure_ready
        with patch('titan_sim.native_plan.extract_plan', side_effect=ValueError('unsupported metadata')) as extract:
            with self.assertRaisesRegex(ValueError, 'after bounded steps completed'):
                export_trainer_plan(trainer, observe_steps=2)
        reference.train_step()
        reference.train_step()
        self.assertEqual(trainer.completed, 2)
        self.assertEqual(trainer.parameter, reference.parameter)
        self.assertEqual(trainer.engine.pp_schedule.ensure_ready, original_ready)
        extract.assert_called_once()

    def test_native_short_run_remains_incomplete(self):
        trainer = self.trainer()
        trainer.completed = 4
        with patch('titan_sim.native_plan.extract_plan', side_effect=self.extract):
            result = export_trainer_plan(trainer, observe_steps=2)
        self.assertEqual(result['capture_status'], 'incomplete')
        self.assertEqual(result['observation']['completed_steps'], 1)

    def test_invalid_mode_and_steps_rejected_before_framework_loading(self):
        for config in ({'recipe_args': []},
                       {'execution': 'fake', 'recipe_args': []},
                       {'execution': 'native', 'recipe_args': [], 'observe_steps': True}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                export_recipe(config)

    def test_unsupported_runtime_does_not_switch_implementations(self):
        trainer = types.SimpleNamespace(engine=types.SimpleNamespace(pp_schedule=object()))
        with self.assertRaisesRegex(ValueError, 'no supported lazy'):
            export_trainer_plan(trainer, observe_steps=1)
