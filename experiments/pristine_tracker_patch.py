"""Runtime patch: swap BundleSdf's tracker-side Python (find_corres /
process_new_frame) for the PRISTINE upstream BundleSDF versions.

Copied verbatim from ~/Desktop/BundleSDF_baseline/bundlesdf.py (upstream
ffa67d4). Our repository's versions diverged in the project's first commit
(e90d327): pairs with too few matches are dropped before RANSAC and their
``_matches`` cleared, ``find_corres`` returns match counts, and the
re-reference loop skips the initial ref frame. The SDF-feedback ablation
(2026-09-02/03) showed our tracker fails to recover after the first pose
write-back while the pristine tracker does, so this patch lets the
experiment driver test that hypothesis without editing main code. The
tracker runs in the main process, so monkeypatching the class works
(the spawned reconstruction process is untouched).
"""

from __future__ import annotations

import logging
import os

import numpy as np

from bundlesdf import BundleSdf, my_cpp, transform_pts


def find_corres(self, frame_pairs):
    logging.info(f"frame_pairs: {len(frame_pairs)}")
    is_match_ref = len(frame_pairs)==1 and frame_pairs[0][0]._ref_frame_id==frame_pairs[0][1]._id and self.bundler._newframe==frame_pairs[0][0]

    imgs, tfs, query_pairs = self.bundler._fm.getProcessedImagePairs(frame_pairs)
    imgs = np.array([np.array(img) for img in imgs])

    if len(query_pairs)==0:
      return

    corres = self.loftr.predict(rgbAs=imgs[::2], rgbBs=imgs[1::2])
    for i_pair in range(len(query_pairs)):
      cur_corres = corres[i_pair][:,:4]
      tfA = np.array(tfs[i_pair*2])
      tfB = np.array(tfs[i_pair*2+1])
      cur_corres[:,:2] = transform_pts(cur_corres[:,:2], np.linalg.inv(tfA))
      cur_corres[:,2:4] = transform_pts(cur_corres[:,2:4], np.linalg.inv(tfB))
      self.bundler._fm._raw_matches[query_pairs[i_pair]] = cur_corres.round().astype(np.uint16)

    min_match_with_ref = self.cfg_track["feature_corres"]["min_match_with_ref"]

    if is_match_ref and len(self.bundler._fm._raw_matches[frame_pairs[0]])<min_match_with_ref:
      self.bundler._fm._raw_matches[frame_pairs[0]] = []
      self.bundler._newframe._status = my_cpp.Frame.FAIL
      logging.info(f'frame {self.bundler._newframe._id_str} mark FAIL, due to no matching')
      return

    self.bundler._fm.rawMatchesToCorres(query_pairs)

    for pair in query_pairs:
      self.bundler._fm.vizCorresBetween(pair[0], pair[1], 'before_ransac')

    self.bundler._fm.runRansacMultiPairGPU(query_pairs)

    for pair in query_pairs:
      self.bundler._fm.vizCorresBetween(pair[0], pair[1], 'after_ransac')


def process_new_frame(self, frame):
    logging.info(f"process frame {frame._id_str}")

    self.bundler._newframe = frame
    os.makedirs(self.debug_dir, exist_ok=True)

    if frame._id>0:
      ref_frame = self.bundler._frames[list(self.bundler._frames.keys())[-1]]
      frame._ref_frame_id = ref_frame._id
      frame._pose_in_model = ref_frame._pose_in_model
    else:
      self.bundler._firstframe = frame

    frame.invalidatePixelsByMask(frame._fg_mask)
    if frame._id==0 and np.abs(np.array(frame._pose_in_model)-np.eye(4)).max()<=1e-4:
      frame.setNewInitCoordinate()


    n_fg = (np.array(frame._fg_mask)>0).sum()
    if n_fg<100:
      logging.info(f"Frame {frame._id_str} cloud is empty, marked FAIL, roi={n_fg}")
      frame._status = my_cpp.Frame.FAIL;
      self.bundler.forgetFrame(frame)
      return

    if self.cfg_track["depth_processing"]["denoise_cloud"]:
      frame.pointCloudDenoise()

    n_valid = frame.countValidPoints()
    n_valid_first = self.bundler._firstframe.countValidPoints()
    if n_valid<n_valid_first/40.0:
      logging.info(f"frame _cloud_down points#: {n_valid} too small compared to first frame points# {n_valid_first}, mark as FAIL")
      frame._status = my_cpp.Frame.FAIL
      self.bundler.forgetFrame(frame)
      return

    if frame._id==0:
      self.bundler.checkAndAddKeyframe(frame)   # First frame is always keyframe
      self.bundler._frames[frame._id] = frame
      return

    min_match_with_ref = self.cfg_track["feature_corres"]["min_match_with_ref"]

    self.find_corres([(frame, ref_frame)])
    matches = self.bundler._fm._matches[(frame, ref_frame)]

    if frame._status==my_cpp.Frame.FAIL:
      logging.info(f"find corres fail, mark {frame._id_str} as FAIL")
      self.bundler.forgetFrame(frame)
      return

    matches = self.bundler._fm._matches[(frame, ref_frame)]
    if len(matches)<min_match_with_ref:
      visibles = []
      for kf in self.bundler._keyframes:
        visible = my_cpp.computeCovisibility(frame, kf)
        visibles.append(visible)
      visibles = np.array(visibles)
      ids = np.argsort(visibles)[::-1]
      found = False
      for id in ids:
        kf = self.bundler._keyframes[id]
        logging.info(f"trying new ref frame {kf._id_str}")
        ref_frame = kf
        frame._ref_frame_id = kf._id
        frame._pose_in_model = kf._pose_in_model
        self.find_corres([(frame, ref_frame)])

        # self.bundler._fm.findCorres(frame, ref_frame)

        if len(self.bundler._fm._matches[(frame,kf)])>=min_match_with_ref:
          logging.info(f"re-choose new ref frame to {kf._id_str}")
          found = True
          break

      if not found:
        frame._status = my_cpp.Frame.FAIL
        logging.info(f"frame {frame._id_str} has not suitable ref_frame, mark as FAIL")
        self.bundler.forgetFrame(frame)
        return

    logging.info(f"frame {frame._id_str} pose update before\n{frame._pose_in_model.round(3)}")
    offset = self.bundler._fm.procrustesByCorrespondence(frame, ref_frame)
    frame._pose_in_model = offset@frame._pose_in_model
    logging.info(f"frame {frame._id_str} pose update after\n{frame._pose_in_model.round(3)}")

    window_size = self.cfg_track["bundle"]["window_size"]
    if len(self.bundler._frames)-len(self.bundler._keyframes)>window_size:
      for k in self.bundler._frames:
        f = self.bundler._frames[k]
        isforget = self.bundler.forgetFrame(f)
        if isforget:
          logging.info(f"exceed window size, forget frame {f._id_str}")
          break

    self.bundler._frames[frame._id] = frame

    self.bundler.selectKeyFramesForBA()

    local_frames = self.bundler._local_frames

    pairs = self.bundler.getFeatureMatchPairs(self.bundler._local_frames)
    self.find_corres(pairs)
    if frame._status==my_cpp.Frame.FAIL:
      self.bundler.forgetFrame(frame)
      return

    find_matches = False
    self.bundler.optimizeGPU(local_frames, find_matches)

    if frame._status==my_cpp.Frame.FAIL:
      self.bundler.forgetFrame(frame)
      return

    self.bundler.checkAndAddKeyframe(frame)


def apply() -> None:
    """Install the pristine tracker functions on BundleSdf (process-wide)."""
    BundleSdf.find_corres = find_corres
    BundleSdf.process_new_frame = process_new_frame
    logging.info("[pristine_tracker_patch] BundleSdf.find_corres / "
                 "process_new_frame replaced with upstream ffa67d4 versions")
