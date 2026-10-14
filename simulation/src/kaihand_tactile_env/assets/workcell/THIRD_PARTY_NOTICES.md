# Third-party notices

## KaiHand model assets

The files under `meshes/` and the following source MJCF fragments were copied
from the local ROS package
`/home/s101/kai_bot_ws/robot_description/kaibot_hand_description`:

- `mjcf/hand_asset.xml`
- `mjcf/hand_contact.xml`
- `mjcf/hand_l_body.xml`
- `mjcf/hand_r_body.xml`
- `mjcf/hands.xml`

That package declares the assets as BSD-licensed in its `package.xml`. The
source tree does not contain a standalone license text, so redistribution
outside the KaiHand project must first confirm the applicable BSD license and
copyright notice with the asset owner.

The local copies intentionally remove the two `thumb_joint4` actuators and
the combined model adds `thumb_joint6 = thumb_joint5` constraints, matching
the 20-motor hardware description in KaiHand's `joint_model.py`.

The right-hand meshes were updated on 2026-08-31 from the segmented CAD export
`KaiBot-Dexhand shell-URDF-R-260831(1530)-tac-p`.  Five closed `*_tac.STL`
pieces are fixed to their existing distal bodies; five open `*_tac_p.STL`
pieces are non-physical probe-placement masks.  Their source hashes and mesh
topology are recorded in
`audits/right_fingertip_masks.audit.json`.
The left distal `link*.STL`, `*_tac.STL` and `*_tac_p.STL` files are local
y-reflected derivatives of the corresponding right shell, closed pad and open
mask.  The left runtime fingertip is the split shell plus a fixed pad geom, so
the replacement preserves the physical fingertip without duplicate collision.
Mirror accuracy and acceptance thresholds are recorded in
`audits/right_to_left_mirror.audit.json`.

## Tianji Marvin M6 assets

The files under `meshes/tianji_m6/` were selected from the Tianji-supplied
`assets.zip` archive.  The combined MJCF uses the corresponding
M6-S-CCS-696-V4 URDF/MJCF transforms, link inertias, limits and effort ranges.
Only the M6 left/right arm meshes and `ZJ_Robot_link.STL` are copied; duplicate
M3, SRS and generated ROS installation files are intentionally omitted.

The supplied `marvin_description/package.xml` declares MIT, but the archive
does not include a standalone copyright/license text.  Confirm redistribution
terms with Tianji before distributing these meshes outside this project.
High-resolution STL files are display geometry; the combined model continues
to use simplified collision proxies and is not a certified safety model.

## Snapshot provenance in this repository

This directory was copied on 2026-08-31 from
`/home/s101/kaihand_teleop_ws/src/kaihand_teleop_sim`.  The workcell scene is
derived from `mjcf/tianji_m6_ccs_kaihands.xml`; the table, two test objects and
fixed cameras are local additions.  Keep this notice with exported datasets
and redistribute the meshes only after the two asset-owner notices above have
been resolved.
