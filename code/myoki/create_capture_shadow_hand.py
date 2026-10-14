"""
Create a screenshot-friendly ShadowHand model without changing the original model.

The generated model keeps the wrist bodies and joints in the kinematic tree so
the existing 24-D ShadowHand qpos mapping remains valid, but removes forearm and
wrist geoms from the visible/collision geometry. It also tightens finger
abduction/adduction limits to +/-5 degrees and uses a white, floorless scene.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HAND_DIR = ROOT / "model" / "mujoco_menagerie" / "shadow_hand"
SRC_HAND = HAND_DIR / "right_hand.xml"
DST_HAND = HAND_DIR / "right_hand_capture.xml"
DST_SCENE = HAND_DIR / "scene_right_capture.xml"

A_A_LIMIT_RAD = 0.0872665  # 5 degrees
INVISIBLE_BODY_NAMES = {"rh_forearm", "rh_wrist"}


def indent(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def remove_forearm_and_wrist_geoms(elem: ET.Element, current_body: str | None = None) -> None:
    if elem.tag == "body":
        current_body = elem.attrib.get("name", current_body)
    if current_body in INVISIBLE_BODY_NAMES:
        for child in list(elem):
            if child.tag != "geom":
                continue
            elem.remove(child)
    for child in list(elem):
        remove_forearm_and_wrist_geoms(child, current_body)


def tighten_abduction_limits(root: ET.Element) -> None:
    limit = f"{-A_A_LIMIT_RAD:.7f} {A_A_LIMIT_RAD:.7f}"
    for default in root.iter("default"):
        if default.attrib.get("class") != "knuckle":
            continue
        for child in default:
            if child.tag == "joint":
                child.set("range", limit)
            elif child.tag == "position":
                child.set("ctrlrange", limit)

    for joint in root.iter("joint"):
        if joint.attrib.get("name") in {"rh_FFJ4", "rh_MFJ4", "rh_RFJ4", "rh_LFJ4"}:
            joint.set("range", limit)


def write_scene() -> None:
    scene = f"""<mujoco model="right_shadow_hand capture scene">
  <include file="right_hand_capture.xml"/>

  <statistic extent="0.24" center="0.02 0 0.105"/>

  <visual>
    <rgba haze="1 1 1 1"/>
    <quality shadowsize="4096"/>
    <global azimuth="220" elevation="-30"/>
    <map fogstart="3" fogend="5"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="512" height="512"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.2" diffuse="0.75 0.75 0.75"/>
    <light pos="0.25 -0.2 1.6" dir="0 0 -1" directional="true" diffuse="0.75 0.75 0.75"/>
  </worldbody>
</mujoco>
"""
    DST_SCENE.write_text(scene, encoding="utf-8")


def main() -> None:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(SRC_HAND, parser=parser)
    root = tree.getroot()
    remove_forearm_and_wrist_geoms(root)
    tighten_abduction_limits(root)
    indent(root)
    ET.ElementTree(root).write(DST_HAND, encoding="utf-8", xml_declaration=False)
    write_scene()
    print(os.fspath(DST_HAND))
    print(os.fspath(DST_SCENE))


if __name__ == "__main__":
    main()
