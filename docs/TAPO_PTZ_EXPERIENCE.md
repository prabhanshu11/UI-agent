# Experience: Tapo C210 PTZ — Pan is Binary, Not Proportional

**Date:** 2026-03-29
**Source project:** `~/Programs/star-trek-camera/`
**Full write-up:** `~/Programs/star-trek-camera/docs/experiences/001_ptz_calibration.md`

## Summary

During Star Trek Camera development, discovered that the Tapo C210's ONVIF ContinuousMove does **not** provide proportional pan control. Pan velocity parameter is ignored — any non-zero pan velocity causes the camera to sweep to the mechanical limit. Tilt works correctly with proportional velocity.

## Relevance to UI-Agent / Star Trek Computer

The ASMP protocol assumes proportional actuator control (`move_continuous(velocity, duration)`). For the Tapo C210 physical camera:
- **Tilt**: Works as ASMP expects (proportional velocity + duration)
- **Pan**: Binary only (full left or full right) — cannot do fine adjustments

This means person tracking via pan requires a different approach:
- Very short ContinuousMove pulses (~50-100ms) to achieve incremental pan
- Or ONVIF RelativeMove (untested)
- Or treat pan as a "room selector" (left position = room A, right = room B)

## Key Lesson

**Build the observation tool first.** A web page with live feed + PTZ buttons revealed the issue in 30 seconds. Automated pixel-diff analysis ran for 45 minutes and produced misleading conclusions. For any actuator-sensor system, real-time human-in-the-loop observation is faster than inference from logged data.
