"""The make() retry classifier must recognize every transient failure mode
observed in the Jul 14-15 production incidents, where the old keyword list
matched none of them and single instance-creation blips zeroed whole
training steps."""
from envs.fleet_env.client import _is_transient_make_error

SEED_PLACEMENT = (
    'Failed to create instance: DriverAPIError: Driver API returned 400: '
    '{"detail":{"error":"seed_placement_failed","reason":"indexer_unreachable",'
    '"message":"could not reach the seed presence indexer after retries; '
    'this is transient - retry the request","retryable":true}}'
)
BURST_CAP = "Too many in-flight instance creations for this team"
RATE_LIMIT = "Rate limit exceeded"
HEALTH = "Instance health check failed"
FATAL_EXAMPLES = (
    "Invalid or missing X-Fleet-Judge-Token",
    "env_key not found: nonexistent-env:v9.9.9",
    'Driver API returned 400: {"detail":{"error":"bad_request","retryable":false}}',
)


def test_retryable_flag_is_transient():
    assert _is_transient_make_error(SEED_PLACEMENT)


def test_burst_cap_is_transient():
    assert _is_transient_make_error(BURST_CAP)


def test_rate_limit_is_transient():
    assert _is_transient_make_error(RATE_LIMIT)


def test_legacy_keywords_still_transient():
    assert _is_transient_make_error(HEALTH)
    assert _is_transient_make_error("Read timeout on connect")


def test_fatal_errors_are_not_transient():
    for msg in FATAL_EXAMPLES:
        assert not _is_transient_make_error(msg), msg
