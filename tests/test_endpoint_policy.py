import pytest

from weightbridge.adaptation.endpoint_policy import EndpointAction, EndpointEvidence, select_endpoint_action
from weightbridge.adaptation.meta_controller import EvidenceState


def state(*, measured=True, headroom=3, budget=10, reachable=True):
    return EvidenceState(measured, headroom, budget, reachable)


def test_anmc_only_sends_actual_endpoint_pass_to_verifier():
    evidence = EndpointEvidence(measured=4, effective=4, surrogate_improvements=3, regressing_probes=2)
    decision = select_endpoint_action(state(), evidence)
    assert (decision.action, decision.reason) == (EndpointAction.CHANGE_OBJECTIVE, "surrogate_endpoint_mismatch")
    assert select_endpoint_action(state(budget=0), evidence).action is EndpointAction.HOLD
    evidence = EndpointEvidence(measured=3, effective=3, practice_passes=1, regressing_probes=1)
    assert select_endpoint_action(state(), evidence).action is EndpointAction.INDEPENDENT_VERIFY


def test_anmc_classifies_evidence_and_does_not_promote():
    empty = EndpointEvidence()
    assert select_endpoint_action(state(measured=False), empty).action is EndpointAction.MEASURE
    assert select_endpoint_action(state(headroom=0), empty).action is EndpointAction.HOLD
    assert select_endpoint_action(state(reachable=None), empty).reason == "reachability_unmeasured"
    assert select_endpoint_action(state(), empty).action is EndpointAction.BOUNDED_PROBE
    assert select_endpoint_action(state(), EndpointEvidence(measured=1)).reason == "ineffective_stored_precision_direction"
    evidence = EndpointEvidence(measured=1, effective=1, regressing_probes=1, smaller_probe_authorized=True)
    assert select_endpoint_action(state(), evidence).action is EndpointAction.SMALLER_PROBE
    for independent, reload, expected in [(False, True, EndpointAction.HOLD), (True, False, EndpointAction.HOLD),
                                           (True, True, EndpointAction.RECORD_VERIFIED)]:
        evidence = EndpointEvidence(measured=1, effective=1, practice_passes=1,
                                    independent_pass=independent, hash_and_reload_verified=reload)
        result = select_endpoint_action(state(), evidence)
        assert result.action is expected
        assert not result.record()["promotion_authority"]
    with pytest.raises(ValueError):
        EndpointEvidence(independent_pass=True)
