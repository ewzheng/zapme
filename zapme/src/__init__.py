"""zapme: vision-driven posture corrector for Raspberry Pi 4.

A camera feed is classified for slouching by the `model` subpackage; on a
sustained positive detection the `runtime` subpackage drives a GPIO line
that gates an EMS unit on the user's back.

See `.llm/llm.MD` for the full project context and `.llm/documentation.MD`
for the binding documentation standard.
"""
