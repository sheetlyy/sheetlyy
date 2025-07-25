import logging
import re
import cv2
import numpy as np
from typing import Optional
from collections import Counter

from worker.transformer.configs import default_config
from worker.transformer.staff2score import Staff2Score
from worker.transformer.parser import TrOMRParser
from worker.utils.constants import NDArray, DURATION_OF_QUARTER
from worker.classes.models import ClefType, Staff
from worker.classes.results import ResultStaff, ResultTimeSignature


logger = logging.getLogger(__name__)


inference: Staff2Score | None = None


def build_image_options(staff_image: NDArray) -> list[NDArray]:
    denoised = cv2.fastNlMeansDenoisingColored(staff_image, None, 10, 10, 7, 21)

    # apply clahe
    gray_image = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_image = clahe.apply(gray_image)
    clahe_applied = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    return [
        staff_image,
        denoised,
        clahe_applied,
    ]


def get_clef_type(result: str) -> Optional[ClefType]:
    match = re.search(r"clef-([A-G])([0-9])", result)
    if match is None:
        return None
    return ClefType(match.group(1))


def flatten_result(result: list[str]) -> list[str]:
    return [
        symbol.split("_")[0].replace("#", "").replace("b", "")
        for group in result
        for symbol in group.split("|")
    ]


def differences(actual: list[str], expected: list[str]) -> int:
    counter1 = Counter(actual)
    counter2 = Counter(expected)
    return sum((counter1 - counter2).values()) + sum((counter2 - counter1).values())


def get_measure_length_variance(result: ResultStaff) -> float:
    durations = [m.length_in_quarters() for m in result.measures]
    return float(np.std(durations - np.mean(durations)))  # type: ignore


def surplus_count(count: int) -> int:
    """
    Assumes that the item should be present at most once.
    """
    return max(count - 1, 0)


def fill_in_time_signature(staff: ResultStaff) -> None:
    avg_measure_length = np.median([m.length_in_quarters() for m in staff.measures])
    for symbol in staff.get_symbols():
        if isinstance(symbol, ResultTimeSignature):
            beat_duration = 4 / symbol.denominator * DURATION_OF_QUARTER
            symbol.numerator = round(avg_measure_length / beat_duration)


def parse_staff_tromr(staff: Staff, staff_image: NDArray) -> Optional[ResultStaff]:
    global inference
    if inference is None:
        inference = Staff2Score(default_config)

    images = [staff_image]
    if len(staff.symbols) > 0:
        images = build_image_options(staff_image)

    notes = staff.get_notes_and_groups()

    best_distance: float = 0
    best_attempt = 0
    best_result: ResultStaff = ResultStaff([])

    for attempt, image in enumerate(images):
        result = inference.predict(image)
        parser = TrOMRParser()
        result_staff = parser.parse_tr_omr_output(str.join("", result))

        clef_type = get_clef_type(result[0])
        if clef_type is None:
            # Returning early is no clef is found is not optimal,
            # but it makes sure that we get a result and it's a corner case,
            # which is not worth the effort to handle right now.
            logger.error(f"Failed to find clef type in {result}")
            return result_staff

        actual = [
            symbol for symbol in result[0].split("+") if symbol.startswith("note")
        ]
        expected = [note.to_tromr_note(clef_type) for note in notes]

        actual = flatten_result(actual)
        expected = flatten_result(expected)

        distance = differences(actual, expected)

        diff_accidentals = abs(
            len(staff.get_accidentals()) - parser.number_of_accidentals()
        )
        measure_length_variance = get_measure_length_variance(result_staff)
        total_structural_elements = (
            surplus_count(parser.number_of_clefs())
            + surplus_count(parser.number_of_key_signatures())
            + surplus_count(parser.number_of_time_signatures())
        )
        total_rating = (
            distance
            + diff_accidentals
            + measure_length_variance
            + total_structural_elements
        ) / max(min(len(expected), len(actual)), 1)

        if best_result.is_empty() or total_rating < best_distance:
            best_distance = total_rating
            best_result = result_staff
            best_attempt = attempt

    fill_in_time_signature(best_result)
    logger.info(
        f"Taking attempt {best_attempt + 1} with distance {best_distance} {best_result}"
    )
    return best_result
