# Spatial Search Protocol - Original User Vision

> Saved verbatim as requested on 2026-01-15

---

In the UI actions library, add this protocol of searching images through a space, searching objects through a space by a camera or anything. The strategy used should be using a 2D version of the binary search and then. That hybridized with the camera or the video feed coming out of the live camera while it rotates that video feeds. Snapshot also being taken, but passively by not stopping the camera. Because usually the snap should be taken when the camera is not in motion so that we can get the highest quality photograph. But in this case, and also if we. If it could be the case that if the camera's internal photo taking algorithm is being used, maybe it applies HDR or something. And then the image produced in its file system is of slightly superior quality. So if we stop the camera and take the photo, then that photo has a higher or those two set of photos have higher chance of being a higher quality. Nevertheless, coming back to the hybrid approach, the approach should be such that while the camera is swiping, it is taking snapshots or like a video and that is also being sent to the some sort of API and those are presented to the agent as hooks. And then if it is detected that on the way it something that keyword was found or something, then not even keyword, the CLAUDE agent just gets that thing. And then the CLAUDE agent is asked to just stop. So in its scale or whatever, there is this point that it stops the feed and then takes a final photograph. And that is the confirmation that it is that the object is found. So this is how we are able to find something the fastest way.

---

## Key Concepts Extracted

1. **2D Binary Search** - Spatial partitioning for efficient object location
2. **Hybrid Continuous/Stopped Capture** - Low-quality while moving, high-quality when stopped
3. **Passive Snapshots** - Continuous capture during motion for coverage
4. **Agent Hooks** - Real-time LLM analysis interrupts sweep when target detected
5. **Confirmation Shot** - Final high-quality capture after stopping
6. **Generalization** - Works for any actuator/sensor pair, not just cameras
