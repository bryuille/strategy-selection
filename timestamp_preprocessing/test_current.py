import unittest
import numpy as np
from current import qualified_looks, raw_roi
from stable_support import ends_for_diameter


def q(t,x=None,y=None,seg=None):
 t=np.array(t,dtype=float);n=len(t)
 return dict(t=t,x=np.zeros(n) if x is None else np.array(x),y=np.zeros(n) if y is None else np.array(y),valid=np.ones(n,bool),segment=np.zeros(n,int) if seg is None else np.array(seg))
H=[5,9,11,5,7,5]
class CurrentTests(unittest.TestCase):
 def test_elapsed_minimum_not_sample_count(self):
  self.assertEqual(len(qualified_looks(q(np.arange(0,75)),H,100,[])),0)
  a=qualified_looks(q(np.arange(0,77,2)),H,100,[]);self.assertEqual(len(a),1);self.assertEqual(a[0]['stable_duration_ms'],76)
 def test_gap_is_boundary(self):
  a=q(np.r_[np.arange(50),np.arange(100,150)],seg=np.r_[np.zeros(50,int),np.ones(50,int)])
  self.assertEqual(qualified_looks(a,H,200,[]),[])
 def test_saccade_is_boundary(self):
  a=qualified_looks(q(np.arange(201)),H,210,[dict(kind='saccade',onset_ms=90,offset_ms=110)])
  self.assertEqual(len(a),2);self.assertEqual([z['stable_duration_ms'] for z in a],[89,89])
 def test_euclidean_not_sum_of_ranges(self):
  t=np.arange(101);a=qualified_looks(q(t,x=1.2*t/100,y=1.2*t/100),H,110,[])
  self.assertEqual(len(a),1)
 def test_drift_can_be_one_look(self):
  t=np.arange(201);a=qualified_looks(q(t,x=t/50),H,210,[])
  self.assertEqual(len(a),1);self.assertAlmostEqual(a[0]['stable_duration_ms'],200)
 def test_invalid_break(self):
  a=q(np.arange(151));a['valid'][75]=False
  self.assertEqual(qualified_looks(a,H,160,[]),[])
 def test_raw_roi_center_priority(self):
  self.assertEqual(raw_roi(2.5,0,H),'center');self.assertEqual(raw_roi(3,0,H),'right')
  self.assertEqual(raw_roi(-5,9,H),'left');self.assertEqual(raw_roi(0,8,H),'unassigned')
 def test_pairwise_windows_bruteforce(self):
  rng=np.random.default_rng(5);xy=rng.normal(size=(30,2));ends,_=ends_for_diameter(xy)
  for i in range(len(xy)):
   j=i
   while j+1<len(xy) and np.max(np.linalg.norm(xy[i:j+2,None]-xy[None,i:j+2],axis=-1))<=2:j+=1
   self.assertEqual(ends[i],j)
if __name__=='__main__':unittest.main()
