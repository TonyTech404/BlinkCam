"""The personalized closed-eye patch bank: descriptors, storage, retrieval.

At setup we harvest patches of the user's own eyes genuinely closed, across a
range of head poses and expressions. At runtime we retrieve the nearest ones
and warp them onto the live eye region.

This is the architecture of EyeOpener (ACM TOG 2016), Bitouk's SIGGRAPH 2008
face swapping, and Photoshop Elements' "Open Closed Eyes", all run backwards.
Our version is easier than any of them because the exemplars come from the same
person, same camera, same room and same session, so identity, sensor and gross
lighting mismatch are eliminated at the source rather than corrected for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import PATCH_H, PATCH_W, extract_patch
from .landmarks import EyeGeometry, FaceFrame

# Descriptor weights. Pose dominates, expression matters more than people
# expect, and scale matters least because the patch is scale-normalised.
WEIGHTS = np.array([
    1.0,    # yaw, degrees / 10
    1.0,    # pitch, degrees / 10
    0.7,    # brow height, an expression proxy
    0.7,    # cheek squint, an expression proxy
    0.3,    # log scale
], dtype=np.float32)


@dataclass
class Descriptor:
    """What a patch is indexed by.

    Roll is absent on purpose: it is compensated by in-plane rotation in the
    patch transform rather than binned, which saves a whole dimension.

    Expression is a first-class dimension, not an afterthought. Smiling,
    talking and squinting all raise the lower lid and change the brow-to-lash
    distance. Indexing on pose alone produces patches that do not match the
    live periorbital shape, and it is the easiest thing here to overlook.
    """

    yaw: float
    pitch: float
    brow_height: float
    cheek_squint: float
    scale: float  # eye width in source pixels

    def vector(self) -> np.ndarray:
        return np.array([
            self.yaw / 10.0,
            self.pitch / 10.0,
            self.brow_height * 10.0,
            self.cheek_squint * 5.0,
            math.log(max(self.scale, 1e-3)),
        ], dtype=np.float32)

    @staticmethod
    def build(face: FaceFrame, eye: EyeGeometry) -> "Descriptor":
        shapes = face.blendshapes
        squint_key = ("cheekSquintRight" if eye.side == "right"
                      else "cheekSquintLeft")
        squint = shapes.get(squint_key, 0.0)
        smile = max(shapes.get("mouthSmileLeft", 0.0),
                    shapes.get("mouthSmileRight", 0.0))
        return Descriptor(
            yaw=face.yaw,
            pitch=face.pitch,
            brow_height=eye.brow_height,
            cheek_squint=max(squint, smile * 0.5),
            scale=eye.width,
        )

    def mirrored(self) -> "Descriptor":
        """The same descriptor as it would read for the opposite eye.

        Faces are approximately bilaterally symmetric, so a patch harvested
        from one eye at yaw +20 can serve the other eye at yaw -20. This
        doubles pose coverage for free.
        """
        return Descriptor(-self.yaw, self.pitch, self.brow_height,
                          self.cheek_squint, self.scale)


@dataclass
class Patch:
    """One harvested closed-eye exemplar."""

    image: np.ndarray  # (PATCH_H, PATCH_W, 3) uint8 BGR, canonical orientation
    descriptor: Descriptor
    side: str  # eye it came from
    annulus: np.ndarray  # mean Lab of the surrounding skin ring
    mirrored: bool = False

    def flipped(self) -> "Patch":
        return Patch(
            image=cv2.flip(self.image, 1),
            descriptor=self.descriptor.mirrored(),
            side="left" if self.side == "right" else "right",
            annulus=self.annulus,
            mirrored=not self.mirrored,
        )


def annulus_mask(shape: tuple[int, int] = (PATCH_H, PATCH_W),
                 inner: float = 0.30, outer: float = 0.46) -> np.ndarray:
    """Ring of periorbital skin surrounding the eye.

    This is the region that exists in BOTH the reference patch and the live
    frame, so it is what illumination and colour matching are estimated from,
    and what candidate patches are scored against.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Normalised elliptical radius about the patch centre.
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    return ((r >= inner * 2) & (r <= outer * 2)).astype(np.float32)


_ANNULUS = annulus_mask()


def annulus_stats(patch: np.ndarray) -> np.ndarray:
    """Mean Lab colour over the periorbital ring."""
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = _ANNULUS[..., None]
    total = float(m.sum())
    if total < 1.0:
        return lab.reshape(-1, 3).mean(axis=0)
    return (lab * m).reshape(-1, 3).sum(axis=0) / total


class PatchBank:
    """Holds harvested patches and retrieves the best matches for a live eye."""

    def __init__(self) -> None:
        self._patches: list[Patch] = []
        self._vectors: np.ndarray | None = None
        # Indices returned by the most recent query, so a caller can feed them
        # back as `sticky` on the next frame.
        self.last_indices: list[int] = []

    def __len__(self) -> int:
        return len(self._patches)

    @property
    def patches(self) -> list[Patch]:
        return self._patches

    def add(self, image: np.ndarray, descriptor: Descriptor, side: str,
            include_mirror: bool = True) -> None:
        patch = Patch(image=image.copy(), descriptor=descriptor, side=side,
                      annulus=annulus_stats(image))
        self._patches.append(patch)
        if include_mirror:
            self._patches.append(patch.flipped())
        self._vectors = None

    def harvest(self, frame: np.ndarray, face: FaceFrame,
                include_mirror: bool = True) -> int:
        """Take a closed-eye patch from each eye of one frame."""
        added = 0
        for eye in face.eyes():
            patch = extract_patch(frame, eye)
            self.add(patch, Descriptor.build(face, eye), eye.side, include_mirror)
            added += 1
        return added

    def _matrix(self) -> np.ndarray:
        if self._vectors is None:
            if not self._patches:
                self._vectors = np.zeros((0, len(WEIGHTS)), np.float32)
            else:
                self._vectors = np.stack(
                    [p.descriptor.vector() for p in self._patches])
        return self._vectors

    def query(self, descriptor: Descriptor, live_patch: np.ndarray,
              k: int = 3, boundary_weight: float = 0.6,
              sticky: "set[int] | None" = None,
              stickiness: float = 0.10
              ) -> list[tuple[Patch, float, float]]:
        """Return up to k candidates as (patch, weight, distance).

        Scoring combines descriptor distance with a boundary-agreement term
        that compares the candidate's surrounding skin ring against the live
        frame's ring. That second term is what stops us selecting a patch
        containing a stray hair or a hand shadow, and it is the step most
        implementations skip. It is Bitouk's "rank by match distance over the
        overlap region", which matters more than pose distance alone.

        Weights are returned for blending the top k rather than hard-selecting
        one, because hard nearest-neighbour pops audibly at bin boundaries.

        `sticky` holds the indices chosen on the previous frame and earns them a
        small discount, which keeps the selection from churning when two
        candidates are nearly tied. Measured churn without it was 11% of frames,
        though the blend weights moved only 0.005 on average, so this is a minor
        refinement rather than the fix for visible instability.
        """
        if not self._patches:
            return []

        vectors = self._matrix()
        query = descriptor.vector()
        pose_d = np.linalg.norm((vectors - query) * WEIGHTS, axis=1)
        if sticky:
            idx = np.fromiter((i for i in sticky if i < len(pose_d)),
                              dtype=np.intp)
            if idx.size:
                pose_d = pose_d.copy()
                pose_d[idx] -= stickiness

        live_annulus = annulus_stats(live_patch)
        cand_annulus = np.stack([p.annulus for p in self._patches])
        # Normalise so overall brightness differences do not dominate; we are
        # asking "does the surrounding skin look like the same neighbourhood",
        # not "is it the same exposure", which relighting handles separately.
        colour_d = np.linalg.norm(cand_annulus - live_annulus, axis=1) / 40.0

        total = pose_d + boundary_weight * colour_d
        k = min(k, len(self._patches))
        idx = np.argpartition(total, k - 1)[:k]
        idx = idx[np.argsort(total[idx])]

        distances = total[idx]
        # Inverse-distance weights, softened so one very close match does not
        # completely dominate and cause popping as ranks swap.
        inv = 1.0 / (distances + 0.25)
        weights = inv / inv.sum()
        self.last_indices = [int(i) for i in idx]
        return [(self._patches[i], float(w), float(d))
                for i, w, d in zip(idx, weights, distances)]

    # ---- persistence ---------------------------------------------------

    def save(self, path: str) -> None:
        if not self._patches:
            raise ValueError("refusing to save an empty bank")
        np.savez_compressed(
            path,
            images=np.stack([p.image for p in self._patches]),
            descriptors=np.stack([p.descriptor.vector() for p in self._patches]),
            raw=np.array([[p.descriptor.yaw, p.descriptor.pitch,
                           p.descriptor.brow_height, p.descriptor.cheek_squint,
                           p.descriptor.scale] for p in self._patches],
                         dtype=np.float32),
            sides=np.array([p.side for p in self._patches]),
            mirrored=np.array([p.mirrored for p in self._patches]),
            annuli=np.stack([p.annulus for p in self._patches]),
        )

    @classmethod
    def load(cls, path: str) -> "PatchBank":
        data = np.load(path, allow_pickle=False)
        bank = cls()
        for image, raw, side, mirrored, annulus in zip(
                data["images"], data["raw"], data["sides"],
                data["mirrored"], data["annuli"]):
            bank._patches.append(Patch(
                image=image,
                descriptor=Descriptor(float(raw[0]), float(raw[1]), float(raw[2]),
                                      float(raw[3]), float(raw[4])),
                side=str(side),
                annulus=annulus,
                mirrored=bool(mirrored),
            ))
        return bank

    def coverage(self, yaw_step: float = 10.0, pitch_step: float = 10.0
                 ) -> dict[tuple[int, int], int]:
        """Patches per pose cell. Shows exactly where the bank has holes."""
        cells: dict[tuple[int, int], int] = {}
        for p in self._patches:
            key = (int(round(p.descriptor.yaw / yaw_step)),
                   int(round(p.descriptor.pitch / pitch_step)))
            cells[key] = cells.get(key, 0) + 1
        return cells
