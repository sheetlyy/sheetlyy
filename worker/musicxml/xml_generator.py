import musicxml.xmlelement.xmlelement as mxl
from typing import Optional

from worker.classes.results import (
    DurationModifier,
    ResultChord,
    ResultClef,
    ResultMeasure,
    ResultNote,
    ResultStaff,
    ResultTimeSignature,
    Page,
)
from worker.utils.constants import DURATION_OF_QUARTER


def build_work() -> mxl.XMLWork:
    work = mxl.XMLWork()
    title = mxl.XMLWorkTitle()
    title._value = "Title"
    work.add_child(title)
    return work


def get_part_id(index: int) -> str:
    return f"P{index + 1}"


def build_part_list(staffs: int) -> mxl.XMLPartList:
    part_list = mxl.XMLPartList()
    for part in range(staffs):
        part_id = get_part_id(part)

        score_part = mxl.XMLScorePart(id=part_id)
        part_name = mxl.XMLPartName(value_="")
        score_part.add_child(part_name)

        score_instrument = mxl.XMLScoreInstrument(id=f"{part_id}-I1")
        instrument_name = mxl.XMLInstrumentName(value_="Piano")
        score_instrument.add_child(instrument_name)
        instrument_sound = mxl.XMLInstrumentSound(value_="keyboard.piano")
        score_instrument.add_child(instrument_sound)
        score_part.add_child(score_instrument)

        midi_instrument = mxl.XMLMidiInstrument(id=f"{part_id}-I1")
        midi_instrument.add_child(mxl.XMLMidiChannel(value_=1))  # type: ignore
        midi_instrument.add_child(mxl.XMLMidiProgram(value_=1))  # type: ignore
        midi_instrument.add_child(mxl.XMLVolume(value_=100))  # type: ignore
        midi_instrument.add_child(mxl.XMLPan(value_=0))  # type: ignore
        score_part.add_child(midi_instrument)

        part_list.add_child(score_part)

    return part_list


def build_or_get_attributes(measure: mxl.XMLMeasure) -> mxl.XMLAttributes:
    for child in measure.get_children_of_type(mxl.XMLAttributes):
        return child

    attributes = mxl.XMLAttributes()
    measure.add_child(attributes)
    return attributes


def build_clef(model_clef: ResultClef, attributes: mxl.XMLAttributes) -> None:
    attributes.add_child(mxl.XMLDivisions(value_=DURATION_OF_QUARTER))  # type: ignore

    key = mxl.XMLKey()
    fifth = mxl.XMLFifths(value_=model_clef.circle_of_fifth)  # type: ignore
    attributes.add_child(key)
    key.add_child(fifth)

    clef = mxl.XMLClef()
    attributes.add_child(clef)

    clef.add_child(mxl.XMLSign(value_=model_clef.clef_type.sign))
    clef.add_child(mxl.XMLLine(value_=model_clef.clef_type.line))  # type: ignore


def build_time_signature(
    model_time_signature: ResultTimeSignature, attributes: mxl.XMLAttributes
) -> None:
    time = mxl.XMLTime()
    attributes.add_child(time)
    time.add_child(mxl.XMLBeats(value_=str(model_time_signature.numerator)))
    time.add_child(mxl.XMLBeatType(value_=str(model_time_signature.denominator)))


def build_rest(model_rest: ResultChord) -> mxl.XMLNote:
    note = mxl.XMLNote()
    note.add_child(mxl.XMLRest(measure="yes"))
    note.add_child(mxl.XMLDuration(value_=model_rest.duration.duration))  # type: ignore
    note.add_child(mxl.XMLType(value_=model_rest.duration.duration_name))
    note.add_child(mxl.XMLStaff(value_=1))  # type: ignore
    return note


def build_note(model_note: ResultNote, is_chord=False) -> mxl.XMLNote:
    note = mxl.XMLNote()
    if is_chord:
        note.add_child(mxl.XMLChord())

    pitch = mxl.XMLPitch()
    model_pitch = model_note.pitch
    pitch.add_child(mxl.XMLStep(value_=model_pitch.step))
    if model_pitch.alter is not None:
        pitch.add_child(mxl.XMLAlter(value_=model_pitch.alter))  # type: ignore
    else:
        pitch.add_child(mxl.XMLAlter(value_=0))  # type: ignore
    pitch.add_child(mxl.XMLOctave(value_=model_pitch.octave))  # type: ignore
    note.add_child(pitch)

    model_duration = model_note.duration
    note.add_child(mxl.XMLType(value_=model_duration.duration_name))
    note.add_child(mxl.XMLDuration(value_=model_duration.duration))  # type: ignore
    note.add_child(mxl.XMLStaff(value_=1))  # type: ignore
    note.add_child(mxl.XMLVoice(value_="1"))

    if model_duration.modifier == DurationModifier.DOT:
        note.add_child(mxl.XMLDot())
    elif model_duration.modifier == DurationModifier.TRIPLET:
        time_modification = mxl.XMLTimeModification()
        time_modification.add_child(mxl.XMLActualNotes(value_=3))  # type: ignore
        time_modification.add_child(mxl.XMLNormalNotes(value_=2))  # type: ignore
        note.add_child(time_modification)

    return note


def build_note_group(note_group: ResultChord) -> list[mxl.XMLNote]:
    result = []
    is_first = True
    for note in note_group.notes:
        result.append(build_note(note, not is_first))
        is_first = False
    return result


def build_chord(chord: ResultChord) -> list[mxl.XMLNote]:
    if chord.is_rest:
        return [build_rest(chord)]
    return build_note_group(chord)


def build_measure(
    measure: ResultMeasure,
    measure_num: int,
    prev_clef: Optional[ResultClef],
    prev_time_sig: Optional[ResultTimeSignature],
) -> tuple[mxl.XMLMeasure, Optional[ResultClef], Optional[ResultTimeSignature]]:
    result = mxl.XMLMeasure(number=str(measure_num))
    is_first_measure = measure_num == 1
    is_new_system = measure.is_new_page or measure.is_new_line

    if measure.is_new_page:
        result.add_child(mxl.XMLPrint(new_page="yes"))
    elif measure.is_new_line and not is_first_measure:
        result.add_child(mxl.XMLPrint(new_system="yes"))

    for symbol in measure.symbols:
        if isinstance(symbol, ResultClef):
            if is_first_measure or is_new_system or symbol != prev_clef:
                attributes = build_or_get_attributes(result)
                build_clef(symbol, attributes)
                prev_clef = symbol

        elif isinstance(symbol, ResultTimeSignature):
            if is_first_measure or is_new_system or symbol != prev_time_sig:
                attributes = build_or_get_attributes(result)
                build_time_signature(symbol, attributes)
                prev_time_sig = symbol

        elif isinstance(symbol, ResultChord):
            for element in build_chord(symbol):
                result.add_child(element)

    return result, prev_clef, prev_time_sig


def build_part(staff: ResultStaff, index: int) -> mxl.XMLPart:
    part = mxl.XMLPart(id=get_part_id(index))
    measure_num = 1
    prev_clef: Optional[ResultClef] = None
    prev_time_sig: Optional[ResultTimeSignature] = None

    for measure in staff.measures:
        xml_measure, prev_clef, prev_time_sig = build_measure(
            measure, measure_num, prev_clef, prev_time_sig
        )
        part.add_child(xml_measure)
        measure_num += 1
    return part


def generate_xml(staffs: list[ResultStaff]) -> mxl.XMLScorePartwise:
    root = mxl.XMLScorePartwise()
    root.add_child(build_work())
    root.add_child(mxl.XMLDefaults())
    root.add_child(build_part_list(len(staffs)))
    for index, staff in enumerate(staffs):
        root.add_child(build_part(staff, index))
    return root
