"""Numerical rejection, fallback, no temporal distortion and legacy parity."""
from dataclasses import replace
from pathlib import Path
import hashlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import forward_width as W
import forward_width_quality as Q
from forward_width_quality_check import continuity, frozen_calibration_input


def model(tail,base,rank):
    return W.PairModel(tail,base,12.,11.5,0.,64.6,np.zeros(3),
                       np.array([.03,0,0]),np.zeros(3),rank)


class QualityTests(unittest.TestCase):
    def test_relocated_frozen_geometry_uses_receiver_files_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'calib').mkdir()
            path=root/'calib/map.json';path.write_bytes(b'original geometry')
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            frozen={'fixed_inputs_sha256':{'/absent/sender/calib/map.json':sha}}
            self.assertEqual(frozen_calibration_input(root,frozen,'calib/map.json'),path.resolve())
            path.write_bytes(b'changed geometry')
            with self.assertRaises(ValueError):frozen_calibration_input(root,frozen,'calib/map.json')

    def test_ambiguous_hashes_and_outside_calibration_paths_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'calib').mkdir();(root/'calib/map.json').write_text('{}')
            frozen={'fixed_inputs_sha256':{'/one/calib/map.json':'one','/two/calib/map.json':'two'}}
            with self.assertRaises(ValueError):frozen_calibration_input(root,frozen,'calib/map.json')
            for path in ['/sender/calib/map.json','calib/../map.json','data/map.json']:
                with self.assertRaises(ValueError):frozen_calibration_input(root,frozen,path)

    def test_low_pixel_residual_does_not_accept_unobservable_width(self):
        prediction = W.PairPrediction(62.6,2.27,3_437_138.)
        decision = Q.select_predictions([model(45,47,1.)],{'pair:45-47':prediction},Q.QualityPolicy(3.35))
        self.assertIsNone(decision.gap_mm)
        self.assertEqual(decision.rejected,{'pair:45-47':'excess_local_sensitivity'})

    def test_bad_best_face_falls_back_before_reference_fusion(self):
        models = [model(45,47,1.),model(49,47,2.),model(44,46,3.)]
        available = {'pair:45-47':W.PairPrediction(62.,.3,1000.),
                     'pair:49-47':W.PairPrediction(40.,.2,1.),
                     'pair:44-46':W.PairPrediction(50.,.2,1.)}
        before = dict(available)
        result = Q.select_predictions(models,available,Q.QualityPolicy(3.35))
        self.assertEqual(result.gap_mm,45.)
        self.assertEqual(set(result.selected_keys),{'pair:49-47','pair:44-46'})
        self.assertEqual(available,before)

    def test_shared_reference_keeps_one_vote_and_legacy_unranked_policy(self):
        models = [model(44,46,3.),model(48,46,1.),model(45,47,2.)]
        available = {m.key:W.PairPrediction(g,.2,1.) for m,g in zip(models,[80.,60.,40.])}
        self.assertEqual(Q.select_predictions(models,available,Q.QualityPolicy(3.35)).gap_mm,50.)
        legacy = [replace(m,holdout_p95_mm=None) for m in models]
        self.assertEqual(Q.select_predictions(legacy,available,Q.QualityPolicy(3.35)).gap_mm,60.)

    def test_boundary_and_noise_floor_are_explicit(self):
        policy = Q.QualityPolicy(3.)
        self.assertIsNone(policy.rejection_reason(W.PairPrediction(100.,.01,10.)))
        self.assertEqual(policy.rejection_reason(W.PairPrediction(100.,.01,10.0001)),
                         'excess_local_sensitivity')

    def test_nonfinite_or_nonpositive_diagnostics_rejected(self):
        policy = Q.QualityPolicy(3.)
        for value in (float('nan'),float('inf')):
            self.assertEqual(policy.rejection_reason(W.PairPrediction(value,.2,1.)),'nonfinite_prediction')
            self.assertEqual(policy.rejection_reason(W.PairPrediction(10.,.2,value)),'nonfinite_prediction')
        for value in (0.,-1.):
            self.assertEqual(policy.rejection_reason(W.PairPrediction(10.,.2,value)),'invalid_sensitivity')
        self.assertEqual(policy.rejection_reason(W.PairPrediction(10.,-.2,1.)),'invalid_sensitivity')
        for value in (0,-1,float('nan'),float('inf'),True,'3'):
            with self.assertRaises(ValueError): Q.QualityPolicy(value)

    def test_no_clipping_and_no_speed_limit_on_well_conditioned_widths(self):
        models = [model(45,47,1.)]
        predicted = [Q.select_predictions(models,{'pair:45-47':W.PairPrediction(g,.2,1.)},
                                          Q.QualityPolicy(3.35)).gap_mm for g in (-5.,90.,10.)]
        self.assertEqual(predicted,[-5.,90.,10.])

    def test_continuity_never_bridges_rejected_or_missing_frames(self):
        result = continuity(np.arange(4)*.038,np.array([10.,np.nan,62.,40.]),np.arange(4))
        self.assertEqual(result['adjacent_valid_pairs'],1)
        self.assertEqual(result['maximum_absolute_step_mm'],22.)
        self.assertEqual(result['largest_steps'][0]['left_frame'],2)

    def test_training_envelope_uses_only_supplied_diagnostics(self):
        training = np.r_[np.arange(1,101)/100,float('nan')]
        self.assertAlmostEqual(Q.derive_policy_limit(training),3*np.percentile(training[:-1],95))
        for scores in ([],np.ones(39),np.r_[np.ones(40),0.]):
            with self.assertRaises(ValueError):Q.derive_policy_limit(scores)

    def test_unknown_or_duplicate_model_keys_rejected(self):
        m=model(45,47,1.)
        with self.assertRaises(ValueError):Q.select_predictions([m,m],{},Q.QualityPolicy(3.35))
        with self.assertRaises(ValueError):Q.select_predictions([m],{'pair:49-47':W.PairPrediction(10.,.2,1.)},Q.QualityPolicy(3.35))


if __name__=='__main__':unittest.main()
