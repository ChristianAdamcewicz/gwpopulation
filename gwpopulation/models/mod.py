def mlow_2_condition(reference_params, mlow_1):
    return dict(
        minimum=reference_params["minimum"],
        maximum=mlow_1
        )

def mu_high_condition(reference_params, mu_low):
    return dict(
        minimum=mu_low,
        maximum=reference_params["maximum"]
        )