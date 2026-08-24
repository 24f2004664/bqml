import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


app = FastAPI()

# Stateful storage for selections.
# A run is stored only after a selection request has been processed.
RUNS = {}

SAFE_INT = 9007199254740991

TIME_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})$"
)

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


# ============================================================
# BASIC HELPERS
# ============================================================

def utf8(value):
    return value.encode("utf-8")


def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sorted_codes(codes):
    return sorted(set(codes), key=utf8)


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def numeric_value(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def safe_nonnegative_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_INT
    )


def safe_positive_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 < value <= SAFE_INT
    )


# ============================================================
# TIMESTAMP HANDLING
# ============================================================

def parse_time(value):
    if not isinstance(value, str):
        return None

    match = TIME_RE.fullmatch(value)

    if match is None:
        return None

    (
        year,
        month,
        day,
        hour,
        minute,
        second,
        fraction,
        offset,
    ) = match.groups()

    year = int(year)
    month = int(month)
    day = int(day)
    hour = int(hour)
    minute = int(minute)
    second = int(second)

    if fraction is None:
        microsecond = 0
    else:
        microsecond = int(fraction.ljust(3, "0")) * 1000

    if offset == "Z":
        tz = timezone.utc
    else:
        sign = 1 if offset[0] == "+" else -1
        offset_hour = int(offset[1:3])
        offset_minute = int(offset[4:6])

        if offset_hour > 14:
            return None

        if offset_minute > 59:
            return None

        # +14:00 and -14:00 are allowed,
        # but +14:01 etc. are not.
        if offset_hour == 14 and offset_minute != 0:
            return None

        tz = timezone(
            sign * timedelta(
                hours=offset_hour,
                minutes=offset_minute,
            )
        )

    try:
        dt = datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            microsecond,
            tzinfo=tz,
        )
    except ValueError:
        return None

    return dt.astimezone(timezone.utc)


def canonical_utc(value):
    dt = parse_time(value)

    if dt is None:
        return None

    return (
        dt.strftime("%Y-%m-%dT%H:%M:%S.")
        + f"{dt.microsecond // 1000:03d}Z"
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def valid_run_id(value):
    return (
        isinstance(value, str)
        and 0 < len(value) <= 128
    )


def valid_selection_row(row):
    if not isinstance(row, dict):
        return False

    if set(row.keys()) != {
        "id",
        "entity",
        "eventTime",
        "predictionTime",
        "version",
        "split",
        "features",
    }:
        return False

    if not isinstance(row["id"], str):
        return False

    if not isinstance(row["entity"], str):
        return False

    if parse_time(row["eventTime"]) is None:
        return False

    if parse_time(row["predictionTime"]) is None:
        return False

    if not safe_nonnegative_int(row["version"]):
        return False

    if row["split"] not in {"TRAIN", "EVAL"}:
        return False

    if not isinstance(row["features"], dict):
        return False

    for feature_name, feature in row["features"].items():
        if not isinstance(feature_name, str):
            return False

        if not isinstance(feature, dict):
            return False

        if set(feature.keys()) != {
            "value",
            "availableAt",
        }:
            return False

        if not isinstance(feature["value"], str):
            return False

        if parse_time(feature["availableAt"]) is None:
            return False

    return True


def valid_trial_shape(trial):
    if not isinstance(trial, dict):
        return False

    if set(trial.keys()) != {
        "trialId",
        "status",
        "evalMetric",
    }:
        return False

    if not safe_nonnegative_int(trial["trialId"]):
        return False

    if trial["status"] not in {
        "SUCCEEDED",
        "FAILED",
    }:
        return False

    # The metric must at least be numeric.
    # A non-finite metric simply makes a SUCCEEDED trial
    # ineligible rather than making the entire selection malformed.
    if not numeric_value(trial["evalMetric"]):
        return False

    return True


def validate_selection_input(body):
    codes = []

    if not valid_run_id(body.get("runId")):
        codes.append("INVALID_INPUT")

    forbidden = body.get("forbiddenFeatures")

    if not isinstance(forbidden, list):
        codes.append("INVALID_INPUT")
    else:
        if any(not isinstance(x, str) for x in forbidden):
            codes.append("INVALID_INPUT")

    limit = body.get("numTrialsLimit")

    if not safe_positive_int(limit):
        codes.append("INVALID_INPUT")

    rows = body.get("rows")

    if not isinstance(rows, list) or len(rows) == 0:
        codes.append("INVALID_INPUT")
    else:
        seen_ids = set()

        for row in rows:
            if not valid_selection_row(row):
                codes.append("INVALID_INPUT")
                continue

            row_id = row["id"]

            if row_id in seen_ids:
                codes.append("INVALID_INPUT")

            seen_ids.add(row_id)

    trials = body.get("trials")

    if not isinstance(trials, list):
        codes.append("INVALID_INPUT")
    else:
        seen_trial_ids = set()

        for trial in trials:
            if not valid_trial_shape(trial):
                codes.append("INVALID_INPUT")
                continue

            trial_id = trial["trialId"]

            if trial_id in seen_trial_ids:
                codes.append("INVALID_INPUT")

            seen_trial_ids.add(trial_id)

    return sorted_codes(codes)


# ============================================================
# DEDUPLICATION
# ============================================================

def dedup_key(row):
    """
    Deduplication key:
        [entity, UTC(eventTime)]
    """
    return (
        row["entity"],
        canonical_utc(row["eventTime"]),
    )


def deduplicate_rows(rows):
    groups = {}

    for row in rows:
        key = dedup_key(row)
        groups.setdefault(key, []).append(row)

    retained = []

    for group in groups.values():
        # Highest version wins.
        # Exact version tie -> UTF-8-smallest ID.
        winner = sorted(
            group,
            key=lambda row: (
                -row["version"],
                utf8(row["id"]),
            ),
        )[0]

        retained.append(winner)

    return retained


# ============================================================
# FEATURE ELIGIBILITY
# ============================================================

def get_feature_names(rows, forbidden):
    """
    A feature is eligible iff:
      - it appears in every retained row
      - it is not forbidden
      - availableAt <= predictionTime in every retained row
    """

    if not rows:
        return []

    common = set(rows[0]["features"].keys())

    for row in rows[1:]:
        common.intersection_update(
            row["features"].keys()
        )

    eligible = []

    for name in common:
        if name in forbidden:
            continue

        available_everywhere = True

        for row in rows:
            available_at = parse_time(
                row["features"][name]["availableAt"]
            )

            prediction_time = parse_time(
                row["predictionTime"]
            )

            if available_at is None:
                available_everywhere = False
                break

            if prediction_time is None:
                available_everywhere = False
                break

            # Point-in-time leakage check.
            if available_at > prediction_time:
                available_everywhere = False
                break

        if available_everywhere:
            eligible.append(name)

    return sorted(eligible, key=utf8)


# ============================================================
# DATASET DIGEST
# ============================================================

def calculate_dataset_digest(
    train_ids,
    eval_ids,
    feature_names,
):
    value = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
    }

    payload = compact_json(value).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


# ============================================================
# TRIAL SELECTION
# ============================================================

def select_trial(trials):
    eligible = []

    for trial in trials:
        if trial["status"] != "SUCCEEDED":
            continue

        if not finite_number(trial["evalMetric"]):
            continue

        eligible.append(trial)

    if not eligible:
        return None

    # Max metric.
    # Exact tie -> smallest integer trialId.
    return sorted(
        eligible,
        key=lambda trial: (
            -float(trial["evalMetric"]),
            trial["trialId"],
        ),
    )[0]


# ============================================================
# SELECTION RESPONSE
# ============================================================

def empty_selection_response(run_id):
    return {
        "runId": run_id,
        "selectedTrialId": None,
        "trainRowIds": [],
        "evalRowIds": [],
        "featureNames": [],
        "datasetDigest": None,
        "reasonCodes": [],
    }


def build_selection(body):
    run_id = body.get("runId")

    result = empty_selection_response(run_id)

    validation_codes = validate_selection_input(body)

    if validation_codes:
        result["reasonCodes"] = validation_codes
        return result

    rows = body["rows"]

    retained = deduplicate_rows(rows)

    forbidden = set(body["forbiddenFeatures"])

    features = get_feature_names(
        retained,
        forbidden,
    )

    train_ids = sorted(
        [
            row["id"]
            for row in retained
            if row["split"] == "TRAIN"
        ],
        key=utf8,
    )

    eval_ids = sorted(
        [
            row["id"]
            for row in retained
            if row["split"] == "EVAL"
        ],
        key=utf8,
    )

    result["trainRowIds"] = train_ids
    result["evalRowIds"] = eval_ids
    result["featureNames"] = features

    # Dataset digest is still well-defined for a valid selection
    # even if trial selection later fails.
    result["datasetDigest"] = calculate_dataset_digest(
        train_ids,
        eval_ids,
        features,
    )

    if len(body["trials"]) > body["numTrialsLimit"]:
        result["reasonCodes"] = [
            "TRIAL_LIMIT_EXCEEDED"
        ]
        return result

    selected = select_trial(body["trials"])

    if selected is None:
        result["reasonCodes"] = [
            "NO_SUCCESSFUL_TRIAL"
        ]
        return result

    result["selectedTrialId"] = selected["trialId"]

    return result


# ============================================================
# RUN ID FINGERPRINT
# ============================================================

def request_fingerprint(body):
    """
    Fingerprint the logical JSON request rather than depending
    on the incoming key order.
    """
    return json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


# ============================================================
# EVALUATION VALIDATION
# ============================================================

def valid_evaluation_input(body):
    if not isinstance(body, dict):
        return False

    required = {
        "phase",
        "runId",
        "selectedTrialId",
        "datasetDigest",
        "metricFloor",
        "requiredSlices",
        "rows",
        "bytesProcessed",
        "maxBytes",
    }

    if set(body.keys()) != required:
        return False

    if body["phase"] != "evaluate":
        return False

    if not valid_run_id(body["runId"]):
        return False

    if not safe_nonnegative_int(
        body["selectedTrialId"]
    ):
        return False

    if not isinstance(body["datasetDigest"], str):
        return False

    if HEX64_RE.fullmatch(
        body["datasetDigest"]
    ) is None:
        return False

    if not finite_number(body["metricFloor"]):
        return False

    if not 0 <= body["metricFloor"] <= 1:
        return False

    required_slices = body["requiredSlices"]

    if not isinstance(required_slices, dict):
        return False

    for name, floor in required_slices.items():
        if not isinstance(name, str):
            return False

        if not finite_number(floor):
            return False

        if not 0 <= floor <= 1:
            return False

    if not isinstance(body["rows"], list):
        return False

    if not safe_nonnegative_int(
        body["bytesProcessed"]
    ):
        return False

    if not safe_nonnegative_int(
        body["maxBytes"]
    ):
        return False

    return True


def valid_test_row(row):
    if not isinstance(row, dict):
        return False

    if set(row.keys()) != {
        "label",
        "prediction",
        "slice",
    }:
        return False

    # bool is an int subclass in Python, so explicitly reject it.
    if (
        not isinstance(row["label"], int)
        or isinstance(row["label"], bool)
        or row["label"] not in (0, 1)
    ):
        return False

    if (
        not isinstance(row["prediction"], int)
        or isinstance(row["prediction"], bool)
        or row["prediction"] not in (0, 1)
    ):
        return False

    if not isinstance(row["slice"], str):
        return False

    if len(row["slice"]) == 0:
        return False

    return True


# ============================================================
# EVALUATION
# ============================================================

def rounded_accuracy(rows):
    if not rows:
        return None

    correct = sum(
        row["label"] == row["prediction"]
        for row in rows
    )

    return round(correct / len(rows), 12)


def build_evaluation(body):
    run_id = body.get("runId")

    result = {
        "runId": run_id,
        "selectedTrialId": body.get(
            "selectedTrialId"
        ),
        "datasetDigest": body.get(
            "datasetDigest"
        ),
        "testMetric": None,
        "criticalSlicePass": False,
        "decision": "reject",
        "bytesProcessed": body.get(
            "bytesProcessed"
        ),
        "reasonCodes": [],
    }

    # --------------------------------------------------------
    # BASIC EVALUATION INPUT
    # --------------------------------------------------------

    if not valid_evaluation_input(body):
        result["reasonCodes"] = [
            "INVALID_INPUT"
        ]
        return result

    codes = []

    # --------------------------------------------------------
    # LINEAGE
    # --------------------------------------------------------

    stored = RUNS.get(run_id)

    lineage_valid = True

    if stored is None:
        lineage_valid = False
    else:
        stored_response = stored["response"]

        # Must be a successful stored selection.
        if stored_response["selectedTrialId"] is None:
            lineage_valid = False

        if (
            stored_response["selectedTrialId"]
            != body["selectedTrialId"]
        ):
            lineage_valid = False

        if (
            stored_response["datasetDigest"]
            != body["datasetDigest"]
        ):
            lineage_valid = False

    if not lineage_valid:
        codes.append("INVALID_LINEAGE")

    # --------------------------------------------------------
    # TEST ROW VALIDATION
    # --------------------------------------------------------

    rows = body["rows"]

    invalid_test_row = False

    for row in rows:
        if not valid_test_row(row):
            invalid_test_row = True
            break

    if invalid_test_row:
        codes.append("INVALID_TEST_ROW")

    # --------------------------------------------------------
    # EMPTY / INVALID TEST DATA
    # --------------------------------------------------------

    if len(rows) == 0 or invalid_test_row:
        # testMetric remains null.
        # Aggregate and slice checks are skipped.
        critical_slice_pass = False

    else:
        # ----------------------------------------------------
        # AGGREGATE ACCURACY
        # ----------------------------------------------------

        test_metric = rounded_accuracy(rows)

        result["testMetric"] = test_metric

        if test_metric < body["metricFloor"]:
            codes.append("AGGREGATE_FLOOR")

        # ----------------------------------------------------
        # SLICE ACCURACY
        # ----------------------------------------------------

        slice_groups = {}

        for row in rows:
            slice_groups.setdefault(
                row["slice"],
                [],
            ).append(row)

        critical_slice_pass = True

        for name, floor in body[
            "requiredSlices"
        ].items():

            if name not in slice_groups:
                codes.append(
                    f"MISSING_SLICE:{name}"
                )
                critical_slice_pass = False
                continue

            accuracy = rounded_accuracy(
                slice_groups[name]
            )

            if accuracy < floor:
                codes.append(
                    f"SLICE_FLOOR:{name}"
                )
                critical_slice_pass = False

    # --------------------------------------------------------
    # BYTE LIMIT
    # --------------------------------------------------------

    if (
        body["bytesProcessed"]
        > body["maxBytes"]
    ):
        codes.append("BYTE_LIMIT")

    # --------------------------------------------------------
    # CRITICAL SLICE FLAG
    # --------------------------------------------------------

    if "INVALID_INPUT" in codes:
        critical_slice_pass = False

    if "INVALID_LINEAGE" in codes:
        critical_slice_pass = False

    if "INVALID_TEST_ROW" in codes:
        critical_slice_pass = False

    if any(
        code.startswith("MISSING_SLICE:")
        or code.startswith("SLICE_FLOOR:")
        for code in codes
    ):
        critical_slice_pass = False

    result["criticalSlicePass"] = (
        critical_slice_pass
    )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    result["reasonCodes"] = sorted_codes(codes)

    if not codes:
        result["decision"] = "admit"
    else:
        result["decision"] = "reject"

    return result


# ============================================================
# HTTP ENDPOINT
# ============================================================

@app.post("/bqml")
async def bqml(request: Request):

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    if not isinstance(body, dict):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    phase = body.get("phase")

    # Unknown or missing phase => HTTP 400 exactly.
    if phase not in {
        "select",
        "evaluate",
    }:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    # ========================================================
    # SELECT
    # ========================================================

    if phase == "select":

        run_id = body.get("runId")

        # Existing valid run ID:
        # identical logical request => exact stored response
        # different request => conflict.
        if (
            isinstance(run_id, str)
            and run_id in RUNS
        ):
            fingerprint = request_fingerprint(
                body
            )

            stored = RUNS[run_id]

            if (
                stored["fingerprint"]
                == fingerprint
            ):
                return JSONResponse(
                    stored["response"]
                )

            return JSONResponse(
                {"error": "RUN_ID_CONFLICT"},
                status_code=409,
            )

        response = build_selection(body)

        # Only a valid runId can be used as state key.
        if valid_run_id(run_id):
            RUNS[run_id] = {
                "fingerprint":
                    request_fingerprint(body),
                "response": response,
            }

        return JSONResponse(response)

    # ========================================================
    # EVALUATE
    # ========================================================

    response = build_evaluation(body)

    return JSONResponse(response)
