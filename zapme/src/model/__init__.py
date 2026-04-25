"""Vision model for posture classification.

This subpackage is intentionally hardware-agnostic. It loads a model,
accepts a frame (numpy / torch tensor), and returns a posture
classification. It must remain importable and unit-testable on a
developer machine without a Pi, a camera, or GPIO.

Do not import from `zapme.src.runtime` here.
"""
