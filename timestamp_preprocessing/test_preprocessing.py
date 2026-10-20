"""Behavioral tests: actual time, missingness, segment boundaries and dynamic selection."""
import unittest,tempfile,csv
from pathlib import Path
import numpy as np
from core import Config,quality,detect,velocity,idt,warp,match,support_weights,segments
from run_preprocessing import ranking

class Checks(unittest.TestCase):
    def test_velocity_uses_actual_elapsed_time(self):
        t=np.cumsum(np.tile([.1,1.9],100));xy=np.column_stack([t*.02,t*-.01])
        v=velocity(t,xy,Config());np.testing.assert_allclose(v[np.isfinite(v).all(axis=1)],[ [20.,-10.] ]*np.isfinite(v).all(axis=1).sum(),atol=1e-8)
        v2=velocity(t*2,xy,Config());np.testing.assert_allclose(v2[np.isfinite(v2).all(axis=1)],[[10.,-5.]]*np.isfinite(v2).all(axis=1).sum(),atol=1e-8)
    def test_fixation_uses_time_not_sample_count(self):
        t=np.arange(0,201,10.);xy=np.zeros((len(t),2))
        self.assertEqual(idt(t,xy,Config()),[(0,20)])
        self.assertEqual(idt(t/10,xy,Config()),[])
    def test_missing_interval_never_bridged(self):
        t=np.r_[np.arange(0,121),np.arange(400,521)]
        x=np.column_stack([np.sin(t)*.01,t/1000]);y=x.copy();p=np.column_stack([np.ones(len(t))*8000,t/1000])
        q=quality(x,y,p,Config());d=detect(q,520,[5]*6,Config())
        self.assertEqual(len(np.unique(d['segment'][d['segment']>=0])),2)
        for e in d['events']:self.assertFalse(e['onset_ms']<200 and e['offset_ms']>300)
        np.testing.assert_array_equal(q['t'],t);np.testing.assert_array_equal(q['x'],x[:,0])
    def test_saccade_threshold_matches_upstream_definition(self):
        from pymovements.events.detection.microsaccades import compute_threshold
        t=np.arange(400.);x=np.column_stack([.02*t+.05*np.sin(t/7),t/1000]);y=np.column_stack([.01*t+.04*np.cos(t/11),t/1000]);p=np.column_stack([np.ones(len(t))*8000,t/1000])
        q=quality(x,y,p,Config());d=detect(q,399,[5]*6,Config())
        v=d['velocity'];v=v[np.isfinite(v).all(axis=1)]
        np.testing.assert_allclose(d['threshold'],8*compute_threshold(v,method='engbert2015'))
    def test_matching_preserves_unmatched(self):
        idx,res=match(np.array([0.,1.,2.]),np.array([0.,2.]),.002)
        np.testing.assert_array_equal(idx,[0,-1,1]);self.assertTrue(np.isnan(res[1]))
    def test_pupil_loss_splits_without_filling(self):
        t=np.arange(300.);x=np.column_stack([t*0,t/1000]);p=np.column_stack([np.full(300,8000.),t/1000]);p[120:150,0]=-32768
        q=quality(x,x,p,Config());self.assertTrue((q['segment'][120:150]==-1).all());np.testing.assert_array_equal(p[120:150,0],np.full(30,-32768.))
    def test_irregular_dwell_not_median_capped(self):
        t=np.array([0.,.05,2.,2.05,4.]);self.assertAlmostEqual(support_weights(t).sum(),4.)
    def test_warp_landmarks(self):
        z=warp(np.array([-3,-3,4,4,0.]),np.array([5,-6,7,-8,7.]),[3,5,6,4,7,8])
        np.testing.assert_allclose(z,[[-1,1],[-1,-1],[1,1],[1,-1],[0,1]])
    def test_dynamic_rank_filters_and_stable_ties(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'r.csv'
            p.write_text('monkey,session,cv_mahalanobis,status\nFaure,F0,99,failed\nFaure,F2,2,passed\nFaure,F1,3,passed\nNielsen,N2,4,passed\nNielsen,N1,4,passed\n')
            self.assertEqual([r['session'] for r in ranking(p)],['F1','F2','N2','N1'])
    def test_duplicate_time_invalid_not_zero_padded(self):
        t=np.array([0.,1.,1.,2.]);xy=np.column_stack([t,t/1000]);p=np.column_stack([np.ones(4)*8000,t/1000]);q=quality(xy,xy,p,Config())
        self.assertTrue((q['flags'][1:3]!=0).all());self.assertEqual(len(q['t']),4)

if __name__=='__main__':unittest.main(verbosity=2)
