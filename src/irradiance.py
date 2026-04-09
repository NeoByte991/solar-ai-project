def estimate_irradiation(hour, clouds):
    if hour < 6 or hour > 18:
        return 0

    peak = 1000  # max sunlight
    base = peak * (1 - abs(hour - 12) / 6)

    # reduce due to clouds
    adjusted = base * (1 - clouds / 100)

    return max(adjusted, 0)