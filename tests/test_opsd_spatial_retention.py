import math
import unittest

import torch

from stable_audio_tools.training.transfusion_opsd.editing_spatial_retention import (
    request_facts, request_spatial_measure, local_covariance, reference_kl,
    execution_authorized_decision,
)
from stable_audio_tools.training.losses.sceneplan_editing_audio import _audio_features, _audio_distances


class SpatialRetentionTests(unittest.TestCase):
    def test_proposed_field_stays_protected_until_execution_confirms_improvement(self):
        decision = {'field': 'start/azimuth_deg', 'position': 25}
        good = [{'plan_index': 1, 'qualified_terminal': True} for _ in range(2)]
        self.assertIsNone(execution_authorized_decision(decision, [], [0., 1.]))
        self.assertIsNone(execution_authorized_decision(decision, good, [1., 0.]))
        self.assertIsNone(execution_authorized_decision(decision, good, [1., 1.]))
        partial = [good[0], {'plan_index': 1, 'qualified_terminal': False}]
        self.assertIsNone(execution_authorized_decision(decision, partial, [0., 1.]))
        self.assertIs(execution_authorized_decision(decision, good, [0., 1.]), decision)

    def test_native_request_interval_and_escaped_content(self):
        text = ('Add the sound described as "A cartoon \\"wah\\" sound.", active from exactly 0.15 seconds '
                'through exactly 0.95 seconds and stationary at azimuth 65 degrees, elevation -11 degrees, '
                'and distance 1.5 meters.')
        facts = request_facts(text, 'event_addition')
        self.assertEqual(facts['activity'], [.15, .95])
        self.assertEqual(facts['fields']['description'], 'A cartoon "wah" sound.')
        self.assertEqual(facts['elevations'], [-11.])
        self.assertEqual(facts['distances'], [1.5])

    def test_frozen_reference_corrects_drift_without_teacher_gradient(self):
        reference = torch.tensor([3., 0., -1.], requires_grad=True)
        holds = [dict(position=1, ids=[0, 1, 2], p=reference.softmax(-1).detach())]
        student = torch.tensor([[0., 4., -1.]], requires_grad=True)
        before = reference_kl(student, holds)
        before.backward()
        self.assertIsNone(reference.grad)
        self.assertLess(student.grad[0, 0], 0.)
        self.assertGreater(student.grad[0, 1], 0.)
        after = reference_kl(student.detach() - student.grad, holds)
        self.assertLess(after, before)

    def test_own_refreshed_reference_has_no_initial_restoring_gradient(self):
        student = torch.tensor([[2., 1., -1.]], requires_grad=True)
        holds = [dict(position=1, ids=[0, 1, 2], p=student[0].softmax(-1).detach())]
        reference_kl(student, holds).backward()
        self.assertLess(float(student.grad.abs().max()), 1e-6)

    def test_foa_spatial_loss_detects_mirror_despite_identical_w_content(self):
        t = torch.arange(17640) / 44100
        w = torch.sin(2 * math.pi * 440 * t)
        truth = torch.stack([w, .7 * w, .1 * w, .5 * w])
        flipped = truth.clone(); flipped[1].neg_(); flipped.requires_grad_(True)
        target_features = _audio_features(truth)
        spec, spatial, _, _ = _audio_distances(_audio_features(flipped), target_features, target_features)
        self.assertLess(float(spec), 1e-6)
        self.assertGreater(float(spatial), .1)
        spatial.backward()
        self.assertTrue(torch.isfinite(flipped.grad).all())
        self.assertGreater(float(flipped.grad[1].norm()), 0.)
        covariance, energy = local_covariance(truth)
        reverse, _ = local_covariance(flipped.detach())
        self.assertGreater(float((covariance - reverse).square().sum()), .1)

    def test_motion_and_elevation_are_measured_over_time(self):
        samples = 44100 * 2
        u = torch.linspace(0, 1, samples)
        endpoints = []
        for az, el in [(-60, -15), (60, 15)]:
            a, e = math.radians(az), math.radians(el)
            endpoints.append(torch.tensor([math.cos(e)*math.cos(a), math.cos(e)*math.sin(a), math.sin(e)]))
        xyz = endpoints[0][:, None] * (1-u) + endpoints[1][:, None] * u
        xyz = xyz / xyz.norm(dim=0, keepdim=True)
        w = torch.sin(2*math.pi*440*torch.arange(samples)/44100)
        wave = torch.stack([w, xyz[1]*w, xyz[2]*w, xyz[0]*w])[None]
        plan = {'sources': [dict(source_id='source_0', kind='music', activity={'onset_sec': 0., 'offset_sec': 2.},
                                 trajectory={'type': 'linear'})]}
        facts = dict(kind='music', azimuths=[-60., 60.], elevations=[-15., 15.], distances=[1., 1.], activity=[0., 2.])
        correct = request_spatial_measure(wave, plan, facts)
        static = wave.clone(); static[:, 1:] = torch.stack([xyz[1, 0]*w, xyz[2, 0]*w, xyz[0, 0]*w])[None]
        wrong = request_spatial_measure(static, plan, facts)
        self.assertTrue(correct['available'])
        self.assertLess(correct['mean_capped_angle_deg'], 1.)
        self.assertGreater(wrong['mean_capped_angle_deg'], 30.)


if __name__ == '__main__':
    unittest.main()
