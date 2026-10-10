import numpy as np
import pytest
from scipy.io import savemat

from ml.annotations import (
    PointAnnotation,
    in_roi,
    load_points_json,
    load_qnrf_mat,
    load_shanghaitech_mat,
    parse_cvat_xml,
    save_points_json,
)


def write_shanghaitech_mat(path, points):
    """Same nesting as the dataset: image_info is a 1x1 cell holding a 1x1 struct."""
    record = np.zeros((1, 1), dtype=[("location", "O"), ("number", "O")])
    record[0, 0]["location"] = np.asarray(points, float)
    record[0, 0]["number"] = len(points)
    cell = np.empty((1, 1), dtype=object)
    cell[0, 0] = record
    savemat(path, {"image_info": cell})


def test_load_shanghaitech_mat(tmp_path):
    points = [[10.5, 20.25], [30.0, 40.0], [55.0, 1.0]]
    write_shanghaitech_mat(tmp_path / "GT_IMG_1.mat", points)
    np.testing.assert_allclose(load_shanghaitech_mat(tmp_path / "GT_IMG_1.mat"), points)


def test_load_qnrf_mat(tmp_path):
    savemat(tmp_path / "img_0001_ann.mat", {"annPoints": np.array([[1.0, 2.0], [3.0, 4.0]])})
    np.testing.assert_allclose(load_qnrf_mat(tmp_path / "img_0001_ann.mat"), [[1, 2], [3, 4]])


def test_points_json_roundtrip(tmp_path):
    path = tmp_path / "annotations" / "vid" / "frame_0001.json"
    save_points_json(path, PointAnnotation("frame_0001.jpg", "vid", 640, 360, np.array([[1.234, 5.678], [9, 10]])))
    ann = load_points_json(path)
    assert (ann.image, ann.video, ann.width, ann.height) == ("frame_0001.jpg", "vid", 640, 360)
    np.testing.assert_allclose(ann.points, [[1.23, 5.68], [9, 10]])
    assert not list(path.parent.glob("*.tmp"))


def test_points_json_roi_roundtrip_and_mask(tmp_path):
    path = tmp_path / "f.json"
    points = np.array([[5.0, 5.0], [15.0, 5.0], [10.0, 20.0]])
    save_points_json(path, PointAnnotation("f.jpg", "v", 40, 30, points, roi=(0, 0, 15, 10)))
    ann = load_points_json(path)
    assert ann.roi == (0, 0, 15, 10)
    assert in_roi(ann.points, ann.roi).tolist() == [True, False, False]  # half-open: x = 15 is outside
    assert in_roi(ann.points, None).all()
    save_points_json(path, PointAnnotation("f.jpg", "v", 40, 30, points))
    assert load_points_json(path).roi is None and "roi" not in path.read_text(encoding="utf-8")


def test_empty_points_json(tmp_path):
    path = tmp_path / "f.json"
    save_points_json(path, PointAnnotation("f.jpg", "v", 10, 10, np.zeros((0, 2))))
    assert load_points_json(path).points.shape == (0, 2)


CVAT_XML = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <image id="0" name="pandharpur_01/frame_0001.jpg" width="1280" height="720">
    <points label="head" source="manual" occluded="0" points="10.5,20.5" z_order="0"/>
    <points label="head" source="manual" occluded="0" points="30,40;50,60" z_order="0"/>
    <points label="other" source="manual" occluded="0" points="1,1" z_order="0"/>
  </image>
  <image id="1" name="frame_0002.jpg" width="1280" height="720"/>
</annotations>
"""


def test_parse_cvat_xml(tmp_path):
    path = tmp_path / "annotations.xml"
    path.write_text(CVAT_XML, encoding="utf-8")
    first, second = parse_cvat_xml(path, label="head")
    assert (first.image, first.video, first.width, first.height) == ("frame_0001.jpg", "pandharpur_01", 1280, 720)
    np.testing.assert_allclose(first.points, [[10.5, 20.5], [30, 40], [50, 60]])
    assert second.video == "" and second.points.shape == (0, 2)
    assert len(parse_cvat_xml(path)[0].points) == 4  # no label filter


def test_cvat_video_export_is_rejected(tmp_path):
    path = tmp_path / "video.xml"
    path.write_text("<annotations><track id='0' label='head'/></annotations>", encoding="utf-8")
    with pytest.raises(ValueError, match="CVAT for images"):
        parse_cvat_xml(path)
