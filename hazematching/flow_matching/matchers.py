from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)


class CCFMFlowMatcher(ExactOptimalTransportConditionalFlowMatcher):
    """ExactOT flow matcher with OT resampling disabled.

    The upstream class resamples (x0, x1) pairs via an OT plan before
    computing the flow. Here that step is skipped — x0 and x1 are used
    exactly as provided, matching the behaviour used in the CCFM paper.
    """

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        return ConditionalFlowMatcher.sample_location_and_conditional_flow(
            self, x0, x1, t, return_noise
        )


class CCFMVariancePreservingFlowMatcher(VariancePreservingConditionalFlowMatcher):
    """Variance-preserving flow matcher with the xt-free compute_conditional_flow signature."""

    def compute_conditional_flow(self, x0, x1, t, xt=None):
        return super().compute_conditional_flow(x0, x1, t)
