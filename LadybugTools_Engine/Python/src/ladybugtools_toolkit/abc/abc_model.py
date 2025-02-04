"""Helper methods to create ABC model configurations."""

import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field  # pylint: disable=no-name-in-module


class HashableBaseModel(BaseModel):
    """A hashable version of the Pydantic BaseModel"""
    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))


class Options(HashableBaseModel):
    """Options for the simulation."""

    csvOutput: bool = Field(
        description="If true, generate CSV output file",
        default=True,
    )
    sensation_adaptation: bool = Field(
        description="If true, use sensation adaptation model",
        default=False,
    )


class ComfortModel(HashableBaseModel):
    """Create a set of comfort model options for the simulation."""

    overall_sensation_model: Literal["original"] = Field(
        description="", default="original"
    )
    local_sensation_model: Literal["original"] = Field(
        description="", default="original"
    )
    overall_comfort_model: Literal["original"] = Field(
        description="", default="original"
    )
    local_comfort_model: Literal["original"] = Field(description="", default="original")


class HeadClothing(HashableBaseModel):
    """Parameters for head clothing insulation and area factor."""

    fclo: float = Field(
        description="Clothing area factor. Unit: fraction.",
        ge=0,
    )
    iclo: float = Field(
        description="Clothing thermal insulation. Unit: clo (1 clo = 1/0.155 W/m2K).",
        ge=0,
    )


class NeckClothing(HashableBaseModel):
    """Parameters for neck clothing insulation and area factor."""

    fclo: float = Field(
        description="Clothing area factor. Unit: fraction.",
        ge=0,
    )
    iclo: float = Field(
        description="Clothing thermal insulation. Unit: clo (1 clo = 1/0.155 W/m2K).",
        ge=0,
    )


class ChestClothing(HashableBaseModel):
    """Parameters for chest clothing insulation and area factor."""

    fclo: float = Field(
        description="Clothing area factor. Unit: fraction.",
        ge=0,
    )
    iclo: float = Field(
        description="Clothing thermal insulation. Unit: clo (1 clo = 1/0.155 W/m2K).",
        ge=0,
    )


class BackClothing(HashableBaseModel):
    """Parameters for back clothing insulation and area factor."""

    fclo: float = Field(
        description="Clothing area factor. Unit: fraction.",
        ge=0,
    )
    iclo: float = Field(
        description="Clothing thermal insulation. Unit: clo (1 clo = 1/0.155 W/m2K).",
        ge=0,
    )


class PelvisClothing(HashableBaseModel):
    """Parameters for pelvis clothing insulation and area factor."""

    fclo: float = Field(
        description="Clothing area factor. Unit: fraction.",
        ge=0,
    )
    iclo: float = Field(
        description="Clothing thermal insulation. Unit: clo (1 clo = 1/0.155 W/m2K).",
        ge=0,
    )


class LeftUpperArmClothing(HashableBaseModel):
    """Parameters for left upper arm clothing insulation and area factor."""

    fclo: float = Field(
        description="Clothing area factor. Unit: fraction.",
        ge=0,
    )
    iclo: float = Field(
        description="Clothing thermal insulation. Unit: clo (1 clo = 1/0.155 W/m2K).",
        ge=0,
    )


class RightUpperArmClothing(HashableBaseModel):
    """Parameters for right upper arm clothing insulation and area factor."""

    fclo: float = Field(
        description="Clothing area factor. Unit: fraction.",
        ge=0,
    )
    iclo: float = Field(
        description="Clothing thermal insulation. Unit: clo (1 clo = 1/0.155 W/m2K).",
        ge=0,
    )


class LeftLowerArmClothing(HashableBaseModel):
    """Parameters for left lower arm clothing insulation and area factor."""

    fclo: float = Field(
        description="Clothing area factor. Unit: fraction.",
        ge=0,
    )
    iclo: float = Field(
        description="Clothing thermal insulation. Unit: clo (1 clo = 1/0.155 W/m2K).",
        ge=0,
    )


class RightLowerArmClothing(HashableBaseModel):
    """Parameters for right lower arm clothing insulation and area factor."""

    fclo: float = Field(
        description="Clothing area factor. Unit: fraction.",
        ge=0,
    )
    iclo: float = Field(
        description="Clothing thermal insulation. Unit: clo (1 clo = 1/0.155 W/m2K).",
        ge=0,
    )


class LeftHandClothing(HashableBaseModel):
    """Parameters for left hand clothing insulation and area factor."""

    fclo: float = Field(
        description="Clothing area factor. Unit: fraction.",
        ge=0,
    )
    iclo: float = Field(
        description="Clothing thermal insulation. Unit: clo (1 clo = 1/0.155 W/m2K).",
        ge=0,
    )


class RightHandClothing(HashableBaseModel):
    """Parameters for right hand clothing insulation and area factor."""

    fclo: float = Field(
        description="Clothing area factor. Unit: fraction.",
        ge=0,
    )
    iclo: float = Field(
        description="Clothing thermal insulation. Unit: clo (1 clo = 1/0.155 W/m2K).",
        ge=0,
    )


class LeftThighClothing(HashableBaseModel):
    """Parameters for left thigh clothing insulation and area factor."""

    fclo: float = Field(
        description="Clothing area factor. Unit: fraction.",
        ge=0,
    )
    iclo: float = Field(
        description="Clothing thermal insulation. Unit: clo (1 clo = 1/0.155 W/m2K).",
        ge=0,
    )


class RightThighClothing(HashableBaseModel):
    """Parameters for right thigh clothing insulation and area factor."""

    fclo: float = Field(
        description="Clothing area factor. Unit: fraction.",
        ge=0,
    )
    iclo: float = Field(
        description="Clothing thermal insulation. Unit: clo (1 clo = 1/0.155 W/m2K).",
        ge=0,
    )


class LeftLowerLegClothing(HashableBaseModel):
    """Parameters for left lower leg clothing insulation and area factor."""

    fclo: float = Field(
        description="Clothing area factor. Unit: fraction.",
        ge=0,
    )
    iclo: float = Field(
        description="Clothing thermal insulation. Unit: clo (1 clo = 1/0.155 W/m2K).",
        ge=0,
    )


class RightLowerLegClothing(HashableBaseModel):
    """Parameters for right lower leg clothing insulation and area factor."""

    fclo: float = Field(
        description="Clothing area factor. Unit: fraction.",
        ge=0,
    )
    iclo: float = Field(
        description="Clothing thermal insulation. Unit: clo (1 clo = 1/0.155 W/m2K).",
        ge=0,
    )


class LeftFootClothing(HashableBaseModel):
    """Parameters for left foot clothing insulation and area factor."""

    fclo: float = Field(
        description="Clothing area factor. Unit: fraction.",
        ge=0,
    )
    iclo: float = Field(
        description="Clothing thermal insulation. Unit: clo (1 clo = 1/0.155 W/m2K).",
        ge=0,
    )


class RightFootClothing(HashableBaseModel):
    """Parameters for right foot clothing insulation and area factor."""

    fclo: float = Field(
        description="Clothing area factor. Unit: fraction.",
        ge=0,
    )
    iclo: float = Field(
        description="Clothing thermal insulation. Unit: clo (1 clo = 1/0.155 W/m2K).",
        ge=0,
    )


class SegmentData(HashableBaseModel):
    """Data for each body part regarding environment conditions."""

    head: HeadClothing = Field(description="The head environment conditions")
    neck: NeckClothing = Field(description="The neck environment conditions")
    chest: ChestClothing = Field(description="The chest environment conditions")
    back: BackClothing = Field(
        description="The back environment conditions",
    )
    pelvis: PelvisClothing = Field(description="The back environment conditions")
    left_upper_arm: LeftUpperArmClothing = Field(
        description="The back environment conditions",
    )
    left_lower_arm: LeftLowerArmClothing = Field(
        description="The back environment conditions",
    )
    left_hand: LeftHandClothing = Field(
        description="The back environment conditions",
    )
    right_upper_arm: RightUpperArmClothing = Field(
        description="The back environment conditions",
    )
    right_lower_arm: RightLowerArmClothing = Field(
        description="The back environment conditions",
    )
    right_hand: RightHandClothing = Field(
        description="The back environment conditions",
    )
    left_thigh: LeftThighClothing = Field(
        description="The back environment conditions",
    )
    left_lower_leg: LeftLowerLegClothing = Field(
        description="The back environment conditions",
    )
    left_foot: LeftFootClothing = Field(
        description="The back environment conditions",
    )
    right_thigh: RightThighClothing = Field(
        description="The back environment conditions",
    )
    right_lower_leg: RightLowerLegClothing = Field(
        description="The back environment conditions",
    )
    right_foot: RightFootClothing = Field(
        description="The back environment conditions",
    )

    def to_dict(self) -> dict:
        """Return a json serialisable version of this object."""
        try:
            d = self.model_dump()
        except AttributeError:
            d = self.dict()

        d["Head"] = d.pop("head")
        d["Neck"] = d.pop("neck")
        d["Chest"] = d.pop("chest")
        d["Back"] = d.pop("back")
        d["Pelvis"] = d.pop("pelvis")
        d["Left Upper Arm"] = d.pop("left_upper_arm")
        d["Left Lower Arm"] = d.pop("left_lower_arm")
        d["Left Hand"] = d.pop("left_hand")
        d["Right Upper Arm"] = d.pop("right_upper_arm")
        d["Right Lower Arm"] = d.pop("right_lower_arm")
        d["Right Hand"] = d.pop("right_hand")
        d["Left Thigh"] = d.pop("left_thigh")
        d["Left Lower Leg"] = d.pop("left_lower_leg")
        d["Left Foot"] = d.pop("left_foot")
        d["Right Thigh"] = d.pop("right_thigh")
        d["Right Lower Leg"] = d.pop("right_lower_leg")
        d["Right Foot"] = d.pop("right_foot")

        return d


class Clothing(HashableBaseModel):
    """Create a set of clothing options for the simulation."""

    ensemble_name: str = Field(description="Name of the ensemble", default="")
    description: str = Field(description="Description of the ensemble", default="")
    segment_data: SegmentData = Field(
        description="Clothing insulation and coverage factors per body part",
    )

    @classmethod
    def nude(cls) -> "Clothing":
        """Return a Clothing object for "Nude" clothing."""
        return cls(
            ensemble_name="Nude",
            description="nude",
            segment_data=SegmentData(
                head=HeadClothing(fclo=1, iclo=0),
                neck=NeckClothing(fclo=1, iclo=0),
                chest=ChestClothing(fclo=1, iclo=0),
                back=BackClothing(fclo=1, iclo=0),
                pelvis=PelvisClothing(fclo=1, iclo=0),
                left_upper_arm=LeftUpperArmClothing(fclo=1, iclo=0),
                right_upper_arm=RightUpperArmClothing(fclo=1, iclo=0),
                left_lower_arm=LeftLowerArmClothing(fclo=1, iclo=0),
                right_lower_arm=RightLowerArmClothing(fclo=1, iclo=0),
                left_hand=LeftHandClothing(fclo=1, iclo=0),
                right_hand=RightHandClothing(fclo=1, iclo=0),
                left_thigh=LeftThighClothing(fclo=1, iclo=0),
                right_thigh=RightThighClothing(fclo=1, iclo=0),
                left_lower_leg=LeftLowerLegClothing(fclo=1, iclo=0),
                right_lower_leg=RightLowerLegClothing(fclo=1, iclo=0),
                left_foot=LeftFootClothing(fclo=1, iclo=0),
                right_foot=RightFootClothing(fclo=1, iclo=0),
            ),
        )

    @classmethod
    def summer_light(cls) -> "Clothing":
        """Return a Clothing object for "Summer light" clothing."""
        return cls(
            ensemble_name="Summer light",
            description="bra+panty, tank top, skirt, sandals",
            segment_data=SegmentData(
                head=HeadClothing(fclo=1, iclo=0),
                neck=NeckClothing(fclo=1, iclo=0),
                chest=ChestClothing(fclo=1.25, iclo=0.83),
                back=BackClothing(fclo=1.07, iclo=0.22),
                pelvis=PelvisClothing(fclo=1.3, iclo=0.99),
                left_upper_arm=LeftUpperArmClothing(fclo=1, iclo=0),
                right_upper_arm=RightUpperArmClothing(fclo=1, iclo=0),
                left_lower_arm=LeftLowerArmClothing(fclo=1, iclo=0),
                right_lower_arm=RightLowerArmClothing(fclo=1, iclo=0),
                left_hand=LeftHandClothing(fclo=1.01, iclo=0.03),
                right_hand=RightHandClothing(fclo=1.01, iclo=0.03),
                left_thigh=LeftThighClothing(fclo=1.26, iclo=0.88),
                right_thigh=RightThighClothing(fclo=1.26, iclo=0.88),
                left_lower_leg=LeftLowerLegClothing(fclo=1.01, iclo=0.05),
                right_lower_leg=RightLowerLegClothing(fclo=1.01, iclo=0.05),
                left_foot=LeftFootClothing(fclo=1.13, iclo=0.44),
                right_foot=RightFootClothing(fclo=1.13, iclo=0.44),
            ),
        )

    @classmethod
    def summer_casual(cls) -> "Clothing":
        """Return a Clothing object for "Summer casual" clothing."""
        return cls(
            ensemble_name="Summer casual",
            description="bra+panty, T-shirt, long pants, socks, sneakers",
            segment_data=SegmentData(
                head=HeadClothing(fclo=1, iclo=0),
                neck=NeckClothing(fclo=1, iclo=0),
                chest=ChestClothing(fclo=1.34, iclo=1.14),
                back=BackClothing(fclo=1.25, iclo=0.84),
                pelvis=PelvisClothing(fclo=1.31, iclo=1.04),
                left_upper_arm=LeftUpperArmClothing(fclo=1.13, iclo=0.42),
                right_upper_arm=RightUpperArmClothing(fclo=1.13, iclo=0.42),
                left_lower_arm=LeftLowerArmClothing(fclo=1, iclo=0),
                right_lower_arm=RightLowerArmClothing(fclo=1, iclo=0),
                left_hand=LeftHandClothing(fclo=1, iclo=0),
                right_hand=RightHandClothing(fclo=1, iclo=0),
                left_thigh=LeftThighClothing(fclo=1.17, iclo=0.58),
                right_thigh=RightThighClothing(fclo=1.17, iclo=0.58),
                left_lower_leg=LeftLowerLegClothing(fclo=1.19, iclo=0.62),
                right_lower_leg=RightLowerLegClothing(fclo=1.19, iclo=0.62),
                left_foot=LeftFootClothing(fclo=1.25, iclo=0.82),
                right_foot=RightFootClothing(fclo=1.25, iclo=0.82),
            ),
        )

    @classmethod
    def summer_business_casual(cls) -> "Clothing":
        """Return a Clothing object for "Summer business casual" clothing."""
        return cls(
            ensemble_name="Summer business casual",
            description="bra+panty, thin dress shirts, long pants, socks, sneakers",
            segment_data=SegmentData(
                head=HeadClothing(fclo=1.03, iclo=0.1),
                neck=NeckClothing(fclo=1.03, iclo=0.1),
                chest=ChestClothing(fclo=1.4, iclo=1.33),
                back=BackClothing(fclo=1.28, iclo=0.93),
                pelvis=PelvisClothing(fclo=1.42, iclo=1.39),
                left_upper_arm=LeftUpperArmClothing(fclo=1.24, iclo=0.79),
                right_upper_arm=RightUpperArmClothing(fclo=1.24, iclo=0.79),
                left_lower_arm=LeftLowerArmClothing(fclo=1.2, iclo=0.66),
                right_lower_arm=RightLowerArmClothing(fclo=1.2, iclo=0.66),
                left_hand=LeftHandClothing(fclo=1.04, iclo=0.13),
                right_hand=RightHandClothing(fclo=1.04, iclo=0.13),
                left_thigh=LeftThighClothing(fclo=1.18, iclo=0.6),
                right_thigh=RightThighClothing(fclo=1.18, iclo=0.6),
                left_lower_leg=LeftLowerLegClothing(fclo=1.17, iclo=0.57),
                right_lower_leg=RightLowerLegClothing(fclo=1.17, iclo=0.57),
                left_foot=LeftFootClothing(fclo=1.23, iclo=0.76),
                right_foot=RightFootClothing(fclo=1.23, iclo=0.76),
            ),
        )

    @classmethod
    def winter_casual(cls) -> "Clothing":
        """Return a Clothing object for "Winter casual" clothing."""
        return cls(
            ensemble_name="Winter casual",
            description="bra+panty, thin dress shirts, long pants, wool sweater, socks, sneakers",
            segment_data=SegmentData(
                head=HeadClothing(fclo=1.03, iclo=0.09),
                neck=NeckClothing(fclo=1.03, iclo=0.09),
                chest=ChestClothing(fclo=1.72, iclo=2.39),
                back=BackClothing(fclo=1.49, iclo=1.64),
                pelvis=PelvisClothing(fclo=1.51, iclo=1.71),
                left_upper_arm=LeftUpperArmClothing(fclo=1.41, iclo=1.36),
                right_upper_arm=RightUpperArmClothing(fclo=1.41, iclo=1.36),
                left_lower_arm=LeftLowerArmClothing(fclo=1.39, iclo=1.29),
                right_lower_arm=RightLowerArmClothing(fclo=1.39, iclo=1.29),
                left_hand=LeftHandClothing(fclo=1.06, iclo=0.21),
                right_hand=RightHandClothing(fclo=1.06, iclo=0.21),
                left_thigh=LeftThighClothing(fclo=1.21, iclo=0.7),
                right_thigh=RightThighClothing(fclo=1.21, iclo=0.7),
                left_lower_leg=LeftLowerLegClothing(fclo=1.16, iclo=0.52),
                right_lower_leg=RightLowerLegClothing(fclo=1.16, iclo=0.52),
                left_foot=LeftFootClothing(fclo=1.23, iclo=0.77),
                right_foot=RightFootClothing(fclo=1.23, iclo=0.77),
            ),
        )

    @classmethod
    def winter_business_formal(cls) -> "Clothing":
        """Return a Clothing object for "Winter business formal" clothing."""
        return cls(
            ensemble_name="Winter business formal",
            description="bra+panty, thin dress shirts, slacks, blazer, \
                tie, belt, socks, formal shoes",
            segment_data=SegmentData(
                head=HeadClothing(fclo=1, iclo=0),
                neck=NeckClothing(fclo=1, iclo=0),
                chest=ChestClothing(fclo=2.08, iclo=3.6),
                back=BackClothing(fclo=1.55, iclo=1.83),
                pelvis=PelvisClothing(fclo=1.51, iclo=1.71),
                left_upper_arm=LeftUpperArmClothing(fclo=1.65, iclo=2.16),
                right_upper_arm=RightUpperArmClothing(fclo=1.65, iclo=2.16),
                left_lower_arm=LeftLowerArmClothing(fclo=1.45, iclo=1.49),
                right_lower_arm=RightLowerArmClothing(fclo=1.45, iclo=1.49),
                left_hand=LeftHandClothing(fclo=1.04, iclo=0.13),
                right_hand=RightHandClothing(fclo=1.04, iclo=0.13),
                left_thigh=LeftThighClothing(fclo=1.19, iclo=0.64),
                right_thigh=RightThighClothing(fclo=1.19, iclo=0.64),
                left_lower_leg=LeftLowerLegClothing(fclo=1.13, iclo=0.43),
                right_lower_leg=RightLowerLegClothing(fclo=1.13, iclo=0.43),
                left_foot=LeftFootClothing(fclo=1.21, iclo=0.69),
                right_foot=RightFootClothing(fclo=1.21, iclo=0.69),
            ),
        )

    @classmethod
    def winter_outerwear(cls) -> "Clothing":
        """Return a Clothing object for "Winter outerwear" clothing."""
        return cls(
            ensemble_name="Winter outerwear",
            description="bra+panty, T-shirt, long sleeve shirts, long pants, \
                winter jacket, socks, sneakers",
            segment_data=SegmentData(
                head=HeadClothing(fclo=1.2, iclo=0.65),
                neck=NeckClothing(fclo=1.2, iclo=0.65),
                chest=ChestClothing(fclo=2.58, iclo=5.26),
                back=BackClothing(fclo=1.92, iclo=3.07),
                pelvis=PelvisClothing(fclo=1.66, iclo=2.2),
                left_upper_arm=LeftUpperArmClothing(fclo=1.94, iclo=3.14),
                right_upper_arm=RightUpperArmClothing(fclo=1.94, iclo=3.14),
                left_lower_arm=LeftLowerArmClothing(fclo=1.62, iclo=2.07),
                right_lower_arm=RightLowerArmClothing(fclo=1.62, iclo=2.07),
                left_hand=LeftHandClothing(fclo=1.02, iclo=0.08),
                right_hand=RightHandClothing(fclo=1.02, iclo=0.08),
                left_thigh=LeftThighClothing(fclo=1.2, iclo=0.67),
                right_thigh=RightThighClothing(fclo=1.2, iclo=0.67),
                left_lower_leg=LeftLowerLegClothing(fclo=1.16, iclo=0.54),
                right_lower_leg=RightLowerLegClothing(fclo=1.16, iclo=0.54),
                left_foot=LeftFootClothing(fclo=1.23, iclo=0.77),
                right_foot=RightFootClothing(fclo=1.23, iclo=0.77),
            ),
        )

    @classmethod
    def from_clo(cls, clo: float) -> "Clothing":
        """Return a Clothing object for a given clo value."""

        return cls(
            ensemble_name=f"{clo} CLO",
            description=f"{clo} CLO (from a single value applied to all body parts)",
            segment_data=SegmentData(
                head=HeadClothing(fclo=1, iclo=clo),
                neck=NeckClothing(fclo=1, iclo=clo),
                chest=ChestClothing(fclo=1, iclo=clo),
                back=BackClothing(fclo=1, iclo=clo),
                pelvis=PelvisClothing(fclo=1, iclo=clo),
                left_upper_arm=LeftUpperArmClothing(fclo=1, iclo=clo),
                right_upper_arm=RightUpperArmClothing(fclo=1, iclo=clo),
                left_lower_arm=LeftLowerArmClothing(fclo=1, iclo=clo),
                right_lower_arm=RightLowerArmClothing(fclo=1, iclo=clo),
                left_hand=LeftHandClothing(fclo=1, iclo=clo),
                right_hand=RightHandClothing(fclo=1, iclo=clo),
                left_thigh=LeftThighClothing(fclo=1, iclo=clo),
                right_thigh=RightThighClothing(fclo=1, iclo=clo),
                left_lower_leg=LeftLowerLegClothing(fclo=1, iclo=clo),
                right_lower_leg=RightLowerLegClothing(fclo=1, iclo=clo),
                left_foot=LeftFootClothing(fclo=1, iclo=clo),
                right_foot=RightFootClothing(fclo=1, iclo=clo),
            ),
        )

    def to_dict(self) -> dict:
        """Helper method to format object as dict."""

        try:
            d = self.model_dump()
        except AttributeError:
            d = self.dict()
        d["segment_data"] = self.segment_data.to_dict()

        return d

    def estimate_clo(self) -> float:
        """Estimate the CLO value of the clothing ensemble."""

        # set body area proportions
        head_area = 0.07
        neck_area = 0.02
        chest_area = 0.09
        back_area = 0.09
        pelvis_area = 0.11
        left_upper_arm_area = 0.05
        right_upper_arm_area = 0.05
        left_lower_arm_area = 0.03
        right_lower_arm_area = 0.03
        left_hand_area = 0.03
        right_hand_area = 0.03
        left_thigh_area = 0.11
        right_thigh_area = 0.11
        left_lower_leg_area = 0.06
        right_lower_leg_area = 0.06
        left_foot_area = 0.03
        right_foot_area = 0.03

        # get weights
        weights = [
            head_area * self.segment_data.head.fclo,
            neck_area * self.segment_data.neck.fclo,
            chest_area * self.segment_data.chest.fclo,
            back_area * self.segment_data.back.fclo,
            pelvis_area * self.segment_data.pelvis.fclo,
            left_upper_arm_area * self.segment_data.left_upper_arm.fclo,
            right_upper_arm_area * self.segment_data.right_upper_arm.fclo,
            left_lower_arm_area * self.segment_data.left_lower_arm.fclo,
            right_lower_arm_area * self.segment_data.right_lower_arm.fclo,
            left_hand_area * self.segment_data.left_hand.fclo,
            right_hand_area * self.segment_data.right_hand.fclo,
            left_thigh_area * self.segment_data.left_thigh.fclo,
            right_thigh_area * self.segment_data.right_thigh.fclo,
            left_lower_leg_area * self.segment_data.left_lower_leg.fclo,
            right_lower_leg_area * self.segment_data.right_lower_leg.fclo,
            left_foot_area * self.segment_data.left_foot.fclo,
            right_foot_area * self.segment_data.right_foot.fclo,
        ]

        # get body part clo values
        clo_values = [
            self.segment_data.head.iclo,
            self.segment_data.neck.iclo,
            self.segment_data.chest.iclo,
            self.segment_data.back.iclo,
            self.segment_data.pelvis.iclo,
            self.segment_data.left_upper_arm.iclo,
            self.segment_data.right_upper_arm.iclo,
            self.segment_data.left_lower_arm.iclo,
            self.segment_data.right_lower_arm.iclo,
            self.segment_data.left_hand.iclo,
            self.segment_data.right_hand.iclo,
            self.segment_data.left_thigh.iclo,
            self.segment_data.right_thigh.iclo,
            self.segment_data.left_lower_leg.iclo,
            self.segment_data.right_lower_leg.iclo,
            self.segment_data.left_foot.iclo,
            self.segment_data.right_foot.iclo,
        ]

        # calculate CLO value using area-weighted average
        estimated_clo = np.average(clo_values, weights=weights)
        return estimated_clo


class BodyBuilder(HashableBaseModel):
    """Body to be simulated. This function customizes the geometry and 
    physiology of the human being modeled based on simple input parameters 
    (such as height, weight, age, gender and so on) to better account for 
    individual differences. See [the paper](https://doi.org/10.1016/S0306-4565(01)00051-1) 
    for more details."""

    age: int = Field(
        description="Age of the person. Unit: years", ge=5, le=100, default=25
    )
    body_fat: float = Field(
        description="Body fat percentage of the person. Unit: fraction",
        ge=0.01,
        le=0.7,
        default=0.13,
    )
    gender: Literal["male", "female"] = Field(
        description="Gender of the person", default="male"
    )
    height: float = Field(
        description="Height of the person. Unit: m", ge=1, le=3, default=1.72
    )
    weight: float = Field(
        description="Weight of the person. Unit: kg", ge=25, le=200, default=74.4
    )
    skin_color: Literal["white", "brown", "black"] = Field(
        description="Skin color of the individual. The solar radiation \
            absorption rate varies depending on this skin color setting. \
                White skin absorbs 62%, brown skin absorbs 70%, and black \
                    skin absorbs 77%.",
        default="brown",
    )


class Phase(HashableBaseModel):
    """A condition describing the environment and the activity of the person."""

    ta: float = Field(description="Dry bulb air temperature. Unit: °C", ge=0, le=50)
    mrt: float = Field(description="Mean radiant temperature. Unit: °C", ge=0, le=50)
    rh: float = Field(description="Relative humidity. Unit: fraction", ge=0, le=1)
    solar: float = Field(description="Solar flux. Unit: W/m2", ge=0, le=10000)
    v: float = Field(description="Air velocity. Unit: m/s", ge=0, le=20)
    met: float = Field(description="Metabolic rate. Unit: met", ge=0.7, le=10)
    met_activity_name: str = Field(description="Name of the activity", repr=False)
    clothing: Clothing = Field(description="Clothing ensemble", repr=False)
    start_time: int = Field(
        description="The start time of the condition being simulated."
    )
    ramp: Optional[bool] = Field(
        default=False,
        description="If true, the condition is ramped between different phases.",
    )
    end_time: int = Field(description="The end time of the condition being simulated.")
    time_units: Optional[Literal["minutes"]] = Field(
        default="minutes",
        description="The units of time for the condition being simulated.",
    )

    @classmethod
    def from_met_clo(
        cls,
        ta: float,
        mrt: float,
        rh: float,
        solar: float,
        v: float,
        met: float,
        clo: float,
        start_time: int,
        end_time: int,
        ramp: bool,
        time_units: str,
    ) -> "Phase":
        """Create a Phase object from MET and CLO values.
        
        Args:
            ta (float):
                Dry bulb air temperature. Unit: °C
            mrt (float):
                Mean radiant temperature. Unit: °C
            rh (float):
                Relative humidity. Unit: fraction
            solar (float):
                Solar flux. Unit: W/m2
            v (float):
                Air velocity. Unit: m/s
            met (float):
                Metabolic rate. Unit: met
            clo (float):
                Clothing insulation. Unit: clo
            start_time (int):
                The start time of the condition being simulated.
            end_time (int):
                The end time of the condition being simulated.
            ramp (bool):
                If true, the condition is ramped between different phases.
            time_units (str):
                The units of time for the condition being simulated.

        Returns:
            Phase: The phase object created from the input values.
        """

        return cls(
            ta=ta,
            mrt=mrt,
            rh=rh,
            solar=solar,
            v=v,
            met=met,
            met_activity_name=f"{met} MET",
            clothing=Clothing.from_clo(clo),
            start_time=start_time,
            end_time=end_time,
            ramp=ramp,
            time_units=time_units,
        )

    def to_dict(self) -> dict:
        """Helper method to format object as dict."""

        try:
            d = self.model_dump()
        except AttributeError:
            d = self.dict()

        default_data = {}
        for key in ["ta", "mrt", "rh", "solar", "v"]:
            default_data[key] = d.pop(key)
        d["default_data"] = default_data

        d["clo_ensemble_name"] = self.clothing.ensemble_name
        d.pop("clothing")

        return d

    def to_series(self) -> pd.Series:
        """Convert the object to a pandas Series."""

        try:
            d = self.model_dump()
        except AttributeError:
            d = self.dict()

        # get clo from clothing
        d["clo"] = self.clothing.estimate_clo()

        # remove unecesary keys
        for k in ["met_activity_name", "clothing"]:
            d.pop(k)

        return pd.Series(d)

class ABCModel(HashableBaseModel):
    """The ABC model configuration."""

    name: str = Field(description="Name of the simulation", repr=True)
    description: str = Field(description="Detailed description of the simulation", repr=True)
    phases: list[Phase] = Field(description="The list of phases in the simulation", repr=True)
    id: Optional[int] = Field(
        description="The unique ID of the simulation. This is not guaranteed to be unique!",
        ge=0,
        default=np.random.randint(100000000),
        repr=True
    )
    reference_time: Optional[datetime] = Field(
        description="The time at which the configuration was created",
        default_factory=datetime.now, repr=False
    )
    output_freq: Optional[int] = Field(
        description="Output frequency, in seconds", default=60, ge=1, repr=False
    )
    options: Optional[Options] = Field(
        description="Simulation options", default_factory=Options, repr=False
    )
    comfort_model: Optional[ComfortModel] = Field(
        description="Comfort model options", default_factory=ComfortModel, repr=False
    )
    bodybuilder: Optional[BodyBuilder] = Field(
        description="The bodybuilder object", default_factory=BodyBuilder, repr=False
    )

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame, **kwargs) -> "ABCModel":
        """Create an ABCModel object from a DataFrame, containing conditions for various phases."""

        if "phases" in kwargs:
            raise ValueError(
                "The 'phases' argument is not allowed when creating an ABCModel from a DataFrame."
            )

        # remove all columns from dataframe that are not in the Phase model
        allowed_columns = list(inspect.signature(Phase.from_met_clo).parameters.keys())

        # check if all columns are present in the dataframe
        for col in allowed_columns:
            if col not in dataframe.columns:
                raise ValueError(f"The column '{col}' is missing.")

        # construct the phases from the dataframe
        kwargs["phases"] = [
            Phase.from_met_clo(**i) for _, i in dataframe[allowed_columns].iterrows()
        ]

        return cls(**kwargs)

    @classmethod
    def from_csv(cls, csv_file: Path, **kwargs) -> "ABCModel":
        """Create an ABCModel object from a CSV file, containing conditions for various phases."""
        return cls.from_dataframe(pd.read_csv(csv_file, header=0), **kwargs)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the object to a pandas DataFrame."""

        ss = []
        for phase in self.phases:
            ss.append(phase.to_series())
        return pd.concat(ss , axis=1).T

    def to_dict(self) -> dict:
        """Convert the object to a dictionary."""

        # ensure each phases time_units is the same
        time_units = set(phase.time_units for phase in self.phases)
        if len(time_units) > 1:
            raise ValueError("All phases must have the same time units.")

        # ensure that each phase is sequential, with the start time of the
        # next phase being the same as the end time of the previous phase
        for i in range(len(self.phases) - 1):
            if self.phases[i].end_time != self.phases[i + 1].start_time:
                raise ValueError(
                    "The end time of one phase must be the same as the \
                        start time of the next phase."
                )

        # construct the root dictionary from the Pydantic models
        try:
            d = self.model_dump()
        except AttributeError:
            d = self.dict()

        # modify the datetime
        d["reference_time"] = d["reference_time"].isoformat()

        # modify clothing parts
        for phase_d, phase in zip(*[d["phases"], self.phases]):
            phase_d["clo_ensemble_name"] = phase.clothing.ensemble_name
            phase_d.pop("clothing")
        clothing_objs = list(set(phase.clothing for phase in self.phases))
        d["clothing"] = [i.to_dict() for i in clothing_objs]

        # modify environment data
        for phase_d, phase in zip(*[d["phases"], self.phases]):
            default_data = {}
            for key in ["ta", "mrt", "rh", "solar", "v"]:
                default_data[key] = phase_d.pop(key)
            phase_d["default_data"] = default_data

        return d

    def to_json(self) -> str:
        """Convert the object to a JSON string."""

        return json.dumps(self.to_dict(), indent=4)

    def to_file(self, file_path: Path) -> Path:
        """Save the object to a JSON file.
        
        """

        if not file_path.suffix == ".json":
            raise ValueError("The file path must have a .json extension.")

        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=4)

        return file_path
