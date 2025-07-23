import logging
from typing import Optional

from app.classes.results import (
    ResultStaff,
    ResultClef,
    ResultChord,
    ResultNote,
    ResultPitch,
)
from app.transformer.circle_of_fifths import get_circle_of_fifth_notes


logger = logging.getLogger(__name__)


def process_note(note: ResultNote, accidentals: dict[str, Optional[int]]) -> None:
    pitch = note.pitch
    note_name = pitch.name_and_octave()

    if pitch.alter is not None:
        accidentals[note_name] = pitch.alter
    elif note_name in accidentals:
        pitch.alter = accidentals[note_name]


def keep_accidentals_until_cancelled(staff: ResultStaff) -> None:
    """
    Implements the rule that accidentals are kept until
    cancelled by a natural sign or a new measure.
    """
    for measure in staff.measures:
        accidentals: dict[str, Optional[int]] = {}
        for symbol in measure.symbols:
            if isinstance(symbol, ResultClef):
                accidentals = {}
            elif isinstance(symbol, ResultChord):
                for note in symbol.notes:
                    process_note(note, accidentals)


def apply_key_to_pitch(
    pitch: ResultPitch, circle_of_fifth: int, circle_of_fifth_notes: list[str]
) -> None:
    if pitch.alter is None:
        altered_by_key = pitch.step in circle_of_fifth_notes
        if altered_by_key:
            pitch.alter = 1 if circle_of_fifth >= 0 else -1


def apply_key_signature(staff: ResultStaff) -> None:
    """
    Applies the key signature to the notes.
    """
    circle_of_fifth = 0
    circle_of_fifth_notes = []
    measure_num = 0
    for measure in staff.measures:
        measure_num += 1
        for symbol in measure.symbols:
            if isinstance(symbol, ResultClef):
                circle_of_fifth = symbol.circle_of_fifth
                circle_of_fifth_notes = get_circle_of_fifth_notes(circle_of_fifth)
            elif isinstance(symbol, ResultChord):
                for note in symbol.notes:
                    apply_key_to_pitch(
                        note.pitch, circle_of_fifth, circle_of_fifth_notes
                    )


def maintain_accidentals(staffs: list[ResultStaff]) -> list[ResultStaff]:
    """
    How MusicXML works: In music XML the alter must be set for every note
    independent of previous alters in the measure or key.

    So a sequence which is printed as
    "Key D Major, Note C, Note Cb, Note C" must be encoded as
    "Key D Major, Note C#, Note Cb, Note Cb" in MusicXML,
    where the first "#" comes from the key of D Major.
    """
    result: list[ResultStaff] = []
    for staff in staffs:
        keep_accidentals_until_cancelled(staff)
        apply_key_signature(staff)
        result.append(staff)
    return result
