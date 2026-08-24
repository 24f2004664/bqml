import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# Stateful storage for this running service.
RUNS = {}

TIME_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})$"
)

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

SAFE_INT = 9007199254740991


def ukey(value):
    return value.encode("utf-8")


def compact(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sorted_codes(codes):
    return sorted(set(codes), key=ukey)


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def safe_nonnegative_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_INT
    )


def parse_time(value):
    if not isinstance(value, str):
        return None

    m = TIME_RE.fullmatch(value)

    if not m:
        return None

    year, month, day, hour, minute, second, fraction, offset = m.groups()

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
        oh = int(offset[1:3])
        om = int(offset[4:6])

        if oh > 14 or om > 59:
            return None

        if oh == 14 and om != 0:
            return None

        tz = timezone(
            sign * timedelta(hours=oh, minutes=om)
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


def utc_time(value):
    dt = parse_time(value)

    if dt is None:
        return None

    return (
        dt.strftime("%Y-%m-%dT%H:%M:%S.")
        + f"{dt.microsecond // 1000:03d}Z"
    )


def valid_string(value, max_len=None, nonempty=False):
    if not isinstance(value, str):
        return False

    if nonempty and len(value) == 0:
        return False

    if max_len is not None and len(value) > max_len:
        return False

    return True


def valid_trial(trial):
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

    if trial["status"] not in {"SUCCEEDED", "FAILED"}:
        return False

    if not finite_number(trial["evalMetric"]):
        return False

    return True


def valid_feature(feature):
    if not isinstance(feature, dict):
        return False

    if set(feature.keys()) != {"value", "availableAt"}:
        return False

    if not isinstance(feature["value"], str):
        return False

    if parse_time(feature["availableAt"]) is None:
        return False

    return True


def valid_selection_row(row):
    if not isinstance(row, dict):
        return False

    required = {
        "id",
        "entity",
        "eventTime",
        "predictionTime",
        "version",
        "split",
        "features",
    }

    if set(row.keys()) != required:
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

    for feature in row["features"].values():
        if not valid_feature(feature):
            return False

    return True


def normalize_text(value):
    value = unicodedata.normalize("NFKC", value)
    return value.strip()


def row_key(row):
    return (
        row["entity"],
        utc_time(row["eventTime"]),
    )


def choose_dedup_winner(rows):
    return sorted(
        rows,
        key=lambda r: (
            -r["version"],
            ukey(r["id"]),
        ),
    )[0]


def feature_names(rows, forbidden):
    if not rows:
        return []

    common = set(rows[0]["features"].keys())

    for row in rows[1:]:
        common &= set(row["features"].keys())

    eligible = []

    for name in common:
        if name in forbidden:
            continue

        ok = True

        for row in rows:
            available = parse_time(
                row["features"][name]["availableAt"]
            )
            prediction = parse_time(row["predictionTime"])

            if available is None or prediction is None:
                ok = False
                break

            if available > prediction:
                ok = False
                break

        if ok:
            eligible.append(name)

    return sorted(eligible, key=ukey)


def dataset_digest(train_ids, eval_ids, features):
    obj = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": features,
    }

    return hashlib.sha256(
        compact(obj).encode("utf-8")
    ).hexdigest()


def selection_fingerprint(body):
    return compact(body)


def base_selection_response(run_id):
    return {
        "runId": run_id,
        "selectedTrialId": None,
        "trainRowIds": [],
        "evalRowIds": [],
        "featureNames": [],
        "datasetDigest": None,
        "reasonCodes": [],
    }


def invalid_selection(run_id, codes):
    result = base_selection_response(run_id)
    result["reasonCodes"] = sorted_codes(codes)
    return result


def valid_run_id(value):
    return (
        isinstance(value, str)
        and len(value) > 0
        and len(value) <= 128
    )


def validate_selection(body):
    codes = []

    run_id = body.get("runId")

    if not valid_run_id(run_id):
        codes.append("INVALID_INPUT")

    if not isinstance(body.get("forbiddenFeatures"), list):
        codes.append("INVALID_INPUT")

    if not safe_positive_int(body.get("numTrialsLimit")):
        codes.append("INVALID_INPUT")

    rows = body.get("rows")

    if not isinstance(rows, list) or len(rows) == 0:
        codes.append("INVALID_INPUT")
    else:
        ids = set()

        for row in rows:
            if not valid_selection_row(row):
                codes.append("INVALID_INPUT")
                continue

            if row["id"] in ids:
                codes.append("INVALID_INPUT")
            ids.add(row["id"])

    trials = body.get("trials")

    if not isinstance(trials, list):
        codes.append("INVALID_INPUT")
    else:
        ids = set()

        for trial in trials:
            if not valid_trial(trial):
                codes.append("INVALID_INPUT")
                continue

            if trial["trialId"] in ids:
                codes.append("INVALID_INPUT")
            ids.add(trial["trialId"])

    return sorted_codes(codes)


def safe_positive_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 < value <= SAFE_INT
    )


def evaluate_selection(body):
    run_id = body.get("runId")
    result = base_selection_response(
        run_id if isinstance(run_id, str) else run_id
    )

    codes = validate_selection(body)

    if codes:
        result["reasonCodes"] = sorted_codes(codes)
        return result

    if len(body["trials"]) > body["numTrialsLimit"]:
        result["reasonCodes"] = ["TRIAL_LIMIT_EXCEEDED"]
        return result

    rows = body["rows"]

    # Deduplicate by [entity, UTC(eventTime)].
    groups = {}

    for row in rows:
        key = row_key(row)
        groups.setdefault(key, []).append(row)

    retained = [
        choose_dedup_winner(group)
        for group in groups.values()
    ]

    # Sort retained rows deterministically by ID.
    retained.sort(key=lambda r: ukey(r["id"]))

    forbidden = set(body["forbiddenFeatures"])

    features = feature_names(retained, forbidden)

    train_ids = sorted(
        [
            row["id"]
            for row in retained
            if row["split"] == "TRAIN"
        ],
        key=ukey,
    )

    eval_ids = sorted(
        [
            row["id"]
            for row in retained
            if row["split"] == "EVAL"
        ],
        key=ukey,
    )

    successful = [
        trial
        for trial in body["trials"]
        if (
            trial["status"] == "SUCCEEDED"
            and finite_number(trial["evalMetric"])
        )
    ]

    if not successful:
        result["trainRowIds"] = train_ids
        result["evalRowIds"] = eval_ids
        result["featureNames"] = features
        result["reasonCodes"] = ["NO_SUCCESSFUL_TRIAL"]
        return result

    selected = max(
        successful,
        key=lambda trial: (
            trial["evalMetric"],
            -trial["trialId"],
        ),
    )

    result["selectedTrialId"] = selected["trialId"]
    result["trainRowIds"] = train_ids
    result["evalRowIds"] = eval_ids
    result["featureNames"] = features

    result["datasetDigest"] = dataset_digest(
        train_ids,
        eval_ids,
        features,
    )

    return result


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

    if not valid_run_id(body["runId"]):
        return False

    if not safe_nonnegative_int(body["selectedTrialId"]):
        return False

    if not isinstance(body["datasetDigest"], str):
        return False

    if HEX64_RE.fullmatch(body["datasetDigest"]) is None:
        return False

    if (
        not finite_number(body["metricFloor"])
        or not 0 <= body["metricFloor"] <= 1
    ):
        return False

    if not isinstance(body["requiredSlices"], dict):
        return False

    for name, floor in body["requiredSlices"].items():
        if (
            not isinstance(name, str)
            or not finite_number(floor)
            or not 0 <= floor <= 1
        ):
            return False

    if not isinstance(body["rows"], list):
        return False

    if not safe_nonnegative_int(body["bytesProcessed"]):
        return False

    if not safe_nonnegative_int(body["maxBytes"]):
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

    if row["label"] not in {0, 1}:
        return False

    if row["prediction"] not in {0, 1}:
        return False

    if not isinstance(row["slice"], str):
        return False

    if len(row["slice"]) == 0:
        return False

    return True


def round12(value):
    return round(value, 12)


def evaluate_phase(body):
    run_id = body.get("runId")

    result = {
        "runId": run_id,
        "selectedTrialId": body.get("selectedTrialId"),
        "datasetDigest": body.get("datasetDigest"),
        "testMetric": None,
        "criticalSlicePass": False,
        "decision": "reject",
        "bytesProcessed": body.get("bytesProcessed"),
        "reasonCodes": [],
    }

    codes = []

    if not valid_evaluation_input(body):
        result["reasonCodes"] = ["INVALID_INPUT"]
        return result

    stored = RUNS.get(run_id)

    if stored is None:
        codes.append("INVALID_LINEAGE")
    else:
        stored_response = stored["response"]

        if (
            stored_response["selectedTrialId"] !=
            body["selectedTrialId"]
            or
            stored_response["datasetDigest"] !=
            body["datasetDigest"]
            or
            stored_response["selectedTrialId"] is None
        ):
            codes.append("INVALID_LINEAGE")

    rows = body["rows"]

    invalid_rows = False

    for row in rows:
        if not valid_test_row(row):
            invalid_rows = True
            break

    if invalid_rows:
        codes.append("INVALID_TEST_ROW")

    # Empty or invalid test rows => no metric/slice evaluation.
    if len(rows) == 0 or invalid_rows:
        critical_pass = False
    else:
        correct = sum(
            row["label"] == row["prediction"]
            for row in rows
        )

        metric = round12(correct / len(rows))
        result["testMetric"] = metric

        if metric < body["metricFloor"]:
            codes.append("AGGREGATE_FLOOR")

        slice_groups = {}

        for row in rows:
            slice_groups.setdefault(row["slice"], []).append(row)

        critical_pass = True

        for name, floor in body["requiredSlices"].items():
            if name not in slice_groups:
                codes.append(f"MISSING_SLICE:{name}")
                critical_pass = False
                continue

            group = slice_groups[name]

            slice_accuracy = round12(
                sum(
                    row["label"] == row["prediction"]
                    for row in group
                ) / len(group)
            )

            if slice_accuracy < floor:
                codes.append(f"SLICE_FLOOR:{name}")
                critical_pass = False

    if body["bytesProcessed"] > body["maxBytes"]:
        codes.append("BYTE_LIMIT")

    # criticalSlicePass must be false for lineage/input/row problems,
    # missing slices, or failed slice floors.
    if "INVALID_LINEAGE" in codes:
        critical_pass = False

    if "INVALID_INPUT" in codes:
        critical_pass = False

    if "INVALID_TEST_ROW" in codes:
        critical_pass = False

    if any(
        code.startswith("MISSING_SLICE:")
        or code.startswith("SLICE_FLOOR:")
        for code in codes
    ):
        critical_pass = False

    result["criticalSlicePass"] = critical_pass

    result["reasonCodes"] = sorted_codes(codes)

    if not codes:
        result["decision"] = "admit"
    else:
        result["decision"] = "reject"

    return result


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

    if phase not in {"select", "evaluate"}:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    if phase == "select":
        run_id = body.get("runId")

        fingerprint = selection_fingerprint(body)

        if isinstance(run_id, str) and run_id in RUNS:
            stored = RUNS[run_id]

            if stored["fingerprint"] == fingerprint:
                return JSONResponse(
                    stored["response"]
                )

            return JSONResponse(
                {"error": "RUN_ID_CONFLICT"},
                status_code=409,
            )

        response = evaluate_selection(body)

        # Persist the complete selection response under runId.
        if isinstance(run_id, str):
            RUNS[run_id] = {
                "fingerprint": fingerprint,
                "response": response,
            }

        return JSONResponse(response)

    response = evaluate_phase(body)

    return JSONResponse(response)
