import logging
import numpy as np
from typing import Optional
from enum import Enum
from dataclasses import dataclass

from worker.utils.constants import TRIPLET_SYMBOL, DURATION_OF_QUARTER

logger = logging.getLogger(__name__)


class ResultSymbol:
    """Base class for all result symbol types."""

    pass


class ClefType:
    @staticmethod
    def treble() -> "ClefType":
        return ClefType(sign="G")

    @staticmethod
    def bass() -> "ClefType":
        return ClefType(sign="F")

    def __init__(self, sign: str) -> None:
        self.sign = sign.upper()
        if self.sign not in ["G", "F", "C"]:
            raise Exception(f"Unknown clef sign {sign}")

        if sign == "G":
            self.line = 2  # treble clef
        if sign == "F":
            self.line = 4  # bass clef
        if sign == "C":
            self.line = 3  # alto clef

    def get_reference_pitch(self) -> "ResultPitch":
        if self.sign == "G":
            g2 = ResultPitch("C", 4, None)
            return g2.move_by(2 * (self.line - 2), None)
        elif self.sign == "F":
            e2 = ResultPitch("E", 2, None)
            return e2.move_by(2 * (self.line - 4), None)
        elif self.sign == "C":
            c3 = ResultPitch("C", 3, None)
            return c3.move_by(2 * (self.line - 3), None)
        raise ValueError(f"Unknown clef sign {str(self)}")

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, ClefType):
            return self.sign == __value.sign and self.line == __value.line
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.sign, self.line))

    def __str__(self) -> str:
        return f"{self.sign}{self.line}"

    def __repr__(self) -> str:
        return str(self)


class ResultTimeSignature(ResultSymbol):
    def __init__(self, numerator: int, denominator: int) -> None:
        self.numerator = numerator
        self.denominator = denominator

    def __eq__(self, __value: object) -> bool:
        return (
            isinstance(__value, ResultTimeSignature)
            and self.numerator == __value.numerator
            and self.denominator == __value.denominator
        )

    def __hash__(self) -> int:
        return hash((self.numerator, self.denominator))

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"

    def __repr__(self) -> str:
        return str(self)


note_names = ["C", "D", "E", "F", "G", "A", "B"]


class ResultPitch:
    def __init__(self, step: str, octave: int, alter: Optional[int]) -> None:
        self.step = step
        self.octave = octave
        self.alter = alter

    def name_and_octave(self) -> str:
        return self.step + str(self.octave)

    def get_relative_position(self, other: "ResultPitch") -> int:
        return (
            (self.octave - other.octave) * 7
            + note_names.index(self.step)
            - note_names.index(other.step)
        )

    def move_by(self, steps: int, alter: Optional[int]) -> "ResultPitch":
        step_idx = (note_names.index(self.step) + steps) % 7
        step = note_names[step_idx]
        octave = self.octave + abs(steps - step_idx) // 6 * np.sign(steps)
        return ResultPitch(step, octave, alter)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, ResultPitch):
            return (
                self.step == __value.step
                and self.octave == __value.octave
                and self.alter == __value.alter
            )
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.step, self.octave, self.alter))

    def alter_str(self) -> str:
        if self.alter == 1:
            return "#"
        elif self.alter == -1:
            return "b"
        elif self.alter == 0:
            return "♮"
        return ""

    def __str__(self) -> str:
        return f"{self.step}{self.octave}{self.alter_str()}"

    def __repr__(self) -> str:
        return str(self)


def get_pitch_from_relative_position(
    reference_pitch: ResultPitch, relative_position: int, alter: Optional[int]
) -> ResultPitch:
    step_index = (note_names.index(reference_pitch.step) + relative_position) % 7
    step = note_names[step_index]

    # abs & sign give us integer division with rounding towards 0
    octave = reference_pitch.octave + abs(
        relative_position - step_index
    ) // 6 * np.sign(relative_position)
    return ResultPitch(step, int(octave), alter)


class ResultClef(ResultSymbol):
    def __init__(self, clef_type: ClefType, circle_of_fifth: int) -> None:
        self.clef_type = clef_type
        self.circle_of_fifth = circle_of_fifth

    def get_reference_pitch(self) -> ResultPitch:
        return self.clef_type.get_reference_pitch()

    def __eq__(self, __value: object) -> bool:
        return (
            isinstance(__value, ResultClef)
            and self.clef_type == __value.clef_type
            and self.circle_of_fifth == __value.circle_of_fifth
        )

    def __hash__(self) -> int:
        return hash((self.clef_type, self.circle_of_fifth))

    def __str__(self) -> str:
        return f"{self.clef_type}/{self.circle_of_fifth}"

    def __repr__(self) -> str:
        return str(self)


def move_pitch_to_clef(
    pitch: ResultPitch, current: Optional[ResultClef], new: ResultClef
) -> ResultPitch:
    """
    Moves the pitch from the current clef to the new clef under the assumption that the clef
    was incorrectly identified, but the pitch position is correct.
    """
    if current is None or new is None or current.clef_type == new.clef_type:
        return pitch
    current_reference_pitch = current.get_reference_pitch()
    new_reference_pitch = new.get_reference_pitch()
    relative_position = pitch.get_relative_position(current_reference_pitch)
    return get_pitch_from_relative_position(
        new_reference_pitch, relative_position, alter=pitch.alter
    )


class DurationModifier(Enum):
    NONE = 0
    DOT = 1
    TRIPLET = 2

    def __init__(self, duration: int) -> None:
        self.duration = duration

    def __str__(self) -> str:
        if self == DurationModifier.NONE:
            return ""
        elif self == DurationModifier.DOT:
            return "."
        elif self == DurationModifier.TRIPLET:
            return TRIPLET_SYMBOL
        else:
            return "Invalid duration"


class ResultDuration:
    def __init__(
        self, base_duration: int, modifier: DurationModifier = DurationModifier.NONE
    ):
        self.base_duration = base_duration
        self.duration = self.adjust_duration(base_duration, modifier)
        self.modifier = modifier
        self.duration_name = self.get_duration_name(base_duration)

    @staticmethod
    def adjust_duration(duration: int, modifier: DurationModifier) -> int:
        if modifier == DurationModifier.DOT:
            return duration * 3 // 2
        elif modifier == DurationModifier.TRIPLET:
            return duration * 2 // 3
        else:
            return duration

    @staticmethod
    def get_duration_name(duration: int) -> str:
        duration_dict = {
            4 * DURATION_OF_QUARTER: "whole",
            2 * DURATION_OF_QUARTER: "half",
            DURATION_OF_QUARTER: "quarter",
            DURATION_OF_QUARTER / 2: "eighth",
            DURATION_OF_QUARTER / 4: "16th",
            DURATION_OF_QUARTER / 8: "32nd",
            DURATION_OF_QUARTER / 16: "64th",
        }
        result = duration_dict.get(duration, None)
        if result is None:
            logger.warning("Unknown duration", duration)
            return "quarter"
        return result

    def __eq__(self, __value: object) -> bool:
        return (
            isinstance(__value, ResultDuration)
            and self.duration == __value.duration
            and self.modifier == __value.modifier
        )

    def __hash__(self) -> int:
        return hash((self.duration, self.modifier))

    def __str__(self) -> str:
        return f"{self.duration_name}{str(self.modifier)}"

    def __repr__(self) -> str:
        return str(self)


class ResultNote:
    def __init__(self, pitch: ResultPitch, duration: ResultDuration):
        self.pitch = pitch
        self.duration = duration

    def __eq__(self, __value: object) -> bool:
        return (
            isinstance(__value, ResultNote)
            and self.pitch == __value.pitch
            and self.duration == __value.duration
        )

    def __hash__(self) -> int:
        return hash((self.pitch, self.duration))

    def __str__(self) -> str:
        return f"{self.pitch}_{self.duration}"

    def __repr__(self) -> str:
        return str(self)


class ResultChord(ResultSymbol):
    """
    A chord which contains 0 to many pitches. 0 pitches indicates that this is a rest.

    The duration of the chord is the distance to the next chord. The individual pitches
    may have a different duration.
    """

    def __init__(self, duration: ResultDuration, notes: list[ResultNote]):
        self.notes = notes
        self.duration = duration

    @property
    def is_rest(self) -> bool:
        return len(self.notes) == 0

    def __eq__(self, __value: object) -> bool:
        return (
            isinstance(__value, ResultChord)
            and self.duration == __value.duration
            and self.notes == __value.notes
        )

    def __hash__(self) -> int:
        return hash((self.notes, self.duration))

    def __str__(self) -> str:
        return f"{'&'.join(map(str, self.notes))}"

    def __repr__(self) -> str:
        return str(self)


class ResultMeasure:
    def __init__(self, symbols: list[ResultSymbol]):
        self.symbols = symbols
        self.is_new_line = False
        self.is_new_page = False

    def is_empty(self) -> bool:
        return len(self.symbols) == 0

    def remove_symbol(self, symbol: ResultSymbol) -> None:
        len_before = len(self.symbols)
        self.symbols = [s for s in self.symbols if s is not symbol]
        if len_before == len(self.symbols):
            raise Exception("Could not remove symbol")

    def length_in_quarters(self) -> float:
        return sum(
            s.duration.duration for s in self.symbols if isinstance(s, ResultChord)
        )

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ResultMeasure):
            return False
        return self.symbols == __value.symbols

    def __hash__(self) -> int:
        return hash(tuple(self.symbols))

    def __str__(self) -> str:
        return f"{' '.join(map(str, self.symbols))}" + "|"

    def __repr__(self) -> str:
        return str(self)


class ResultStaff:
    def __init__(self, measures: list[ResultMeasure]):
        self.measures = measures

    def merge(self, other: "ResultStaff") -> "ResultStaff":
        return ResultStaff(self.measures + other.measures)

    def get_symbols(self) -> list[ResultSymbol]:
        return [symbol for measure in self.measures for symbol in measure.symbols]

    def is_empty(self) -> bool:
        return len(self.measures) == 0

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ResultStaff):
            return False
        return self.measures == __value.measures

    def __hash__(self) -> int:
        return hash(tuple(self.measures))

    def __str__(self) -> str:
        return f"Staff({' '.join(map(str, self.measures))})"

    def __repr__(self) -> str:
        return str(self)


@dataclass
class Page:
    staffs: list[ResultStaff]
