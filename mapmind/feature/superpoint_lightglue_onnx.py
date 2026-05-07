import argparse
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


FEATURE_DIR = Path(__file__).resolve().parent
DEFAULT_SUPERPOINT_MODEL = (
    FEATURE_DIR / "model" / "superpoint_onnx" / "1" / "superpoint_fp16.onnx"
)
DEFAULT_LIGHTGLUE_MODEL = (
    FEATURE_DIR / "model" / "lightglue_onnx" / "1" / "lightglue_fp16.onnx"
)


def _select_providers(providers: Optional[Iterable[str]] = None):
    requested = list(providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
    available = ort.get_available_providers()
    selected = [provider for provider in requested if provider in available]
    return selected or ["CPUExecutionProvider"]


def _load_session(model_path: Path, providers: Optional[Iterable[str]] = None):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError("ONNX model not found: {}".format(model_path))
    return ort.InferenceSession(str(model_path), providers=_select_providers(providers))


def _as_batched_keypoints(kpts: np.ndarray) -> np.ndarray:
    kpts = np.asarray(kpts, dtype=np.float32)
    if kpts.ndim == 2:
        kpts = kpts[None]
    if kpts.ndim != 3 or kpts.shape[-1] != 2:
        raise ValueError("Expected keypoints shape [1, N, 2], got {}".format(kpts.shape))
    return kpts


def _as_batched_descriptors(desc: np.ndarray) -> np.ndarray:
    desc = np.asarray(desc, dtype=np.float32)
    if desc.ndim == 2:
        desc = desc[None]
    if desc.ndim != 3 or desc.shape[-1] != 256:
        raise ValueError("Expected descriptors shape [1, N, 256], got {}".format(desc.shape))
    return desc


def _as_batched_scores(scores: Optional[np.ndarray], length: int) -> np.ndarray:
    if scores is None:
        return np.arange(length, 0, -1, dtype=np.float32)[None]
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim == 1:
        scores = scores[None]
    if scores.ndim != 2:
        raise ValueError("Expected scores shape [1, N], got {}".format(scores.shape))
    return scores


class SuperPointONNX:
    def __init__(
        self,
        model_path: Path = DEFAULT_SUPERPOINT_MODEL,
        max_image_shape: int = 960,
        keypoint_thresh: float = 0.015,
        providers: Optional[Iterable[str]] = None,
    ):
        self.session = _load_session(model_path, providers)
        self.max_image_shape = max_image_shape
        self.keypoint_thresh = keypoint_thresh
        self.output_names = [output.name for output in self.session.get_outputs()]

    def _preprocess(self, image_numpy: np.ndarray):
        image = (
            cv2.cvtColor(image_numpy, cv2.COLOR_BGR2GRAY)
            if image_numpy.ndim == 3
            else image_numpy.copy()
        )
        image_size = (image.shape[1], image.shape[0])

        if image.shape[1] > self.max_image_shape:
            new_height = int(self.max_image_shape * image.shape[0] / image.shape[1])
            image_size = (self.max_image_shape, new_height)
            image = cv2.resize(image, image_size).astype(np.uint8)
        if image.shape[0] > self.max_image_shape:
            new_width = int(self.max_image_shape * image.shape[1] / image.shape[0])
            image_size = (new_width, self.max_image_shape)
            image = cv2.resize(image, image_size).astype(np.uint8)

        image = np.expand_dims(image, axis=(0, 1)).astype(np.uint8)
        return image, image_size

    def run(self, image_numpy: np.ndarray):
        if image_numpy is None:
            raise ValueError("image_numpy is None")

        image, image_size = self._preprocess(image_numpy)
        outputs = self.session.run(
            self.output_names,
            {
                "image": image,
                "keypoint_threshold": np.array([[self.keypoint_thresh]], dtype=np.float32),
            },
        )
        result = dict(zip(self.output_names, outputs))

        kpts = _as_batched_keypoints(result["kpts"])
        descps = _as_batched_descriptors(result["descps"])
        scores = _as_batched_scores(result["scores"], kpts.shape[1])

        factor_x = image_numpy.shape[1] / image_size[0]
        factor_y = image_numpy.shape[0] / image_size[1]
        kpts = kpts.copy()
        kpts[..., 0] *= factor_x
        kpts[..., 1] *= factor_y

        return kpts.astype(np.float32), descps.astype(np.float32), scores.astype(np.float32)


class LightGlueONNX:
    def __init__(
        self,
        model_path: Path = DEFAULT_LIGHTGLUE_MODEL,
        match_thresh: float = 0.1,
        max_num_keypoints: int = 512,
        providers: Optional[Iterable[str]] = None,
    ):
        self.session = _load_session(model_path, providers)
        self.match_thresh = match_thresh
        self.max_num_keypoints = max_num_keypoints
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.score_name = "score" if "score" in self.output_names else "match_scores"

    def _pad_or_truncate(
        self,
        kpts: np.ndarray,
        desc: np.ndarray,
        scores: Optional[np.ndarray] = None,
    ):
        kpts = _as_batched_keypoints(kpts)[0]
        desc = _as_batched_descriptors(desc)[0]
        scores = _as_batched_scores(scores, len(kpts))[0]

        if len(kpts) != len(desc):
            raise ValueError(
                "Keypoints and descriptors length mismatch: {} vs {}".format(
                    len(kpts), len(desc)
                )
            )

        original_count = len(kpts)
        order = np.arange(len(kpts), dtype=np.int32)
        if len(kpts) > self.max_num_keypoints:
            keep = np.argsort(scores)[-self.max_num_keypoints :][::-1]
            kpts = kpts[keep]
            desc = desc[keep]
            order = order[keep]

        valid_count = len(kpts)
        padded_kpts = np.zeros((1, self.max_num_keypoints, 2), dtype=np.float32)
        padded_desc = np.zeros((1, self.max_num_keypoints, 256), dtype=np.float32)
        mask = np.zeros((1, self.max_num_keypoints, 1), dtype=bool)

        if valid_count > 0:
            padded_kpts[0, :valid_count] = kpts
            padded_desc[0, :valid_count] = desc
            mask[0, :valid_count, 0] = True

        return padded_kpts, padded_desc, mask, order, valid_count, original_count

    def run(
        self,
        kpts0: np.ndarray,
        desc0: np.ndarray,
        img_shape0: np.ndarray,
        kpts1: np.ndarray,
        desc1: np.ndarray,
        img_shape1: np.ndarray,
        scores0: Optional[np.ndarray] = None,
        scores1: Optional[np.ndarray] = None,
    ):
        kpts0_pad, desc0_pad, mask0, order0, valid0, original_count0 = self._pad_or_truncate(
            kpts0, desc0, scores0
        )
        kpts1_pad, desc1_pad, mask1, order1, valid1, _ = self._pad_or_truncate(
            kpts1, desc1, scores1
        )

        img_shape0 = np.asarray(img_shape0, dtype=np.int32).reshape(1, 2)
        img_shape1 = np.asarray(img_shape1, dtype=np.int32).reshape(1, 2)

        outputs = self.session.run(
            self.output_names,
            {
                "kpts0": kpts0_pad,
                "kpts1": kpts1_pad,
                "desc0": desc0_pad,
                "desc1": desc1_pad,
                "mask0": mask0,
                "mask1": mask1,
                "img_shape0": img_shape0,
                "img_shape1": img_shape1,
                "threshold": np.array([[self.match_thresh]], dtype=np.float32),
            },
        )
        result = dict(zip(self.output_names, outputs))

        match_indices = np.asarray(result["match_indices"][0], dtype=np.int32)
        match_scores = np.asarray(result[self.score_name][0], dtype=np.float32)

        # Convert matches from padded/truncated LightGlue indices back to original indices.
        original_indices = np.full(valid0, -1, dtype=np.int32)
        original_scores = match_scores[:valid0].copy()
        for i in range(valid0):
            matched = match_indices[i]
            if 0 <= matched < valid1:
                original_indices[i] = order1[matched]

        restored_indices = np.full(original_count0, -1, dtype=np.int32)
        restored_scores = np.zeros(original_count0, dtype=np.float32)
        restored_indices[order0[:valid0]] = original_indices
        restored_scores[order0[:valid0]] = original_scores
        return restored_indices, restored_scores


class SuperPointLightGlueMatcher:
    def __init__(
        self,
        superpoint_model: Path = DEFAULT_SUPERPOINT_MODEL,
        lightglue_model: Path = DEFAULT_LIGHTGLUE_MODEL,
        max_image_shape: int = 960,
        keypoint_thresh: float = 0.015,
        match_thresh: float = 0.1,
        providers: Optional[Iterable[str]] = None,
    ):
        self.superpoint = SuperPointONNX(
            superpoint_model,
            max_image_shape=max_image_shape,
            keypoint_thresh=keypoint_thresh,
            providers=providers,
        )
        self.lightglue = LightGlueONNX(
            lightglue_model,
            match_thresh=match_thresh,
            providers=providers,
        )

    def match(self, image0: np.ndarray, image1: np.ndarray) -> Dict[str, np.ndarray]:
        kpts0, desc0, scores0 = self.superpoint.run(image0)
        kpts1, desc1, scores1 = self.superpoint.run(image1)
        img_shape0 = np.array([[image0.shape[0], image0.shape[1]]], dtype=np.int32)
        img_shape1 = np.array([[image1.shape[0], image1.shape[1]]], dtype=np.int32)

        match_indices, match_scores = self.lightglue.run(
            kpts0,
            desc0,
            img_shape0,
            kpts1,
            desc1,
            img_shape1,
            scores0=scores0,
            scores1=scores1,
        )
        valid = match_indices >= 0
        matches = np.column_stack((np.where(valid)[0], match_indices[valid]))

        return {
            "keypoints0": kpts0[0],
            "keypoints1": kpts1[0],
            "scores0": scores0[0],
            "scores1": scores1[0],
            "matches0": match_indices,
            "matching_scores0": match_scores,
            "matches": matches.astype(np.int32),
            "scores": match_scores[valid],
        }


def draw_matches(
    img0: np.ndarray,
    img1: np.ndarray,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    match_indices: np.ndarray,
    match_scores: np.ndarray,
    max_display: int = 5000,
    score_thresh: float = 0.0,
) -> np.ndarray:
    img0_color = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR) if img0.ndim == 2 else img0.copy()
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1.copy()

    h0, w0 = img0_color.shape[:2]
    h1, w1 = img1_color.shape[:2]
    out_img = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    out_img[:h0, :w0] = img0_color
    out_img[:h1, w0:] = img1_color

    matches = [
        (i, match_indices[i], match_scores[i])
        for i in range(len(match_indices))
        if match_indices[i] >= 0 and match_scores[i] >= score_thresh
    ]
    matches = sorted(matches, key=lambda item: -item[2])[:max_display]

    for i, j, score in matches:
        pt0 = tuple(map(int, kpts0[i]))
        pt1 = tuple(map(int, kpts1[j] + np.array([w0, 0], dtype=np.float32)))
        color = (0, int(score * 255), 255 - int(score * 255))
        cv2.circle(out_img, pt0, 5, color, -1)
        cv2.circle(out_img, pt1, 5, color, -1)
        cv2.line(out_img, pt0, pt1, color, 2)

    return out_img


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run local SuperPoint + LightGlue ONNX image matching."
    )
    parser.add_argument("image0", help="First image path")
    parser.add_argument("image1", help="Second image path")
    parser.add_argument(
        "--output",
        default="mapmind/feature/lightglue_matches.jpg",
        help="Output visualization path",
    )
    parser.add_argument("--max-image-shape", type=int, default=960)
    parser.add_argument("--keypoint-thresh", type=float, default=0.015)
    parser.add_argument("--match-thresh", type=float, default=0.1)
    parser.add_argument(
        "--provider",
        action="append",
        dest="providers",
        help="ONNXRuntime provider, can be passed multiple times",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    image0 = cv2.imread(args.image0)
    image1 = cv2.imread(args.image1)
    if image0 is None:
        raise FileNotFoundError("Could not read image: {}".format(args.image0))
    if image1 is None:
        raise FileNotFoundError("Could not read image: {}".format(args.image1))

    matcher = SuperPointLightGlueMatcher(
        max_image_shape=args.max_image_shape,
        keypoint_thresh=args.keypoint_thresh,
        match_thresh=args.match_thresh,
        providers=args.providers,
    )

    start_ms = time.time() * 1000
    result = matcher.match(image0, image1)
    elapsed_ms = time.time() * 1000 - start_ms

    vis = draw_matches(
        image0,
        image1,
        result["keypoints0"],
        result["keypoints1"],
        result["matches0"],
        result["matching_scores0"],
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)

    print("keypoints0:", result["keypoints0"].shape)
    print("keypoints1:", result["keypoints1"].shape)
    print("matches:", result["matches"].shape[0])
    print("time_ms: {:.2f}".format(elapsed_ms))
    print("output:", output_path)


if __name__ == "__main__":
    main()
