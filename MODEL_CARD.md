# AI Golf Coach Model Card

**Author:** Isaiah Goh  
**Mentor:** Dr. Qingyang Xiao

## Intended use

This repository is an educational prototype demonstrating video pose estimation, interpretable biomechanics features, supervised machine learning, a multi-layer neural network, feedback-based recommendations, and downloadable reports.

## Not intended for

- Medical diagnosis, rehabilitation, or injury-risk prediction
- Certified coaching decisions or player selection
- Autonomous high-stakes recommendations
- Claims of real-world model accuracy without additional validation

## Model components

1. **MediaPipe pose estimation:** estimates up to 33 body landmarks from sampled frames.
2. **Transparent scoring:** uses camera-dependent geometric indicators such as joint angles, torso tilt, sway, rotation, hand speed, and tempo.
3. **Random Forest:** trained on synthetic feature rows and transparent synthetic labels to demonstrate supervised-learning integration.
4. **Deep neural network:** a three-hidden-layer MLP trained on the same synthetic architecture-demonstration dataset.
5. **Contextual bandit:** updates drill preferences from a user's 1–5 usefulness rating.

## Data limitations

The repository does not include a coach-labeled golf dataset. Synthetic scores must not be reported as validated golf-coaching accuracy. A production study should use consented, de-identified data, multiple coaches, subject-level train/test separation, diverse users and camera views, and preregistered outcome metrics.

## Technical limitations

- Single-camera 2D pose is sensitive to viewpoint and occlusion.
- The clubhead, ball flight, pressure, and ground-reaction forces are not tracked.
- The swing phase detector is heuristic.
- Streamlit Community Cloud has shared resource constraints, so videos are truncated and frame-sampled.
- Feedback learning can encode user preference bias.
