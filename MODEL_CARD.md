# Model Card — AI Golf Coach Educational Prototype

## Overview

The app uses MediaPipe Pose Landmarker in the visitor's browser to estimate 33 body landmarks from sampled video frames. It derives 2D geometric features and creates educational swing metrics. When the model cannot load or pose coverage is too low, the app uses a native motion-centroid fallback.

## Intended use

- Student AI/ML demonstration
- Golf movement visualization
- Practice reflection and repeatable camera-based comparison
- Exploration of supervised features, neural-network architecture concepts, and feedback learning

## Not intended for

- Medical diagnosis or rehabilitation decisions
- Injury prediction
- Professional certification of golf technique
- Autonomous high-stakes decisions
- Comparing people for employment, insurance, education, or eligibility

## Inputs

- Browser-decodable golf video
- Optional golfer name, handedness, experience level, and practice goal
- Optional Streamlit profile information in a separate session form

## Outputs

- Estimated pose landmarks or motion-centroid data
- 2D elbow and knee angles
- Torso tilt, shoulder/hip orientation difference, head motion, hip sway, hand-speed proxy
- Swing phases and educational scores
- Coaching suggestions and a four-week practice direction
- JSON/CSV/PNG/WebM downloads

## AI methods represented

1. **Feature engineering / supervised-ML preparation:** frame-level landmarks and derived features can form a labeled training table.
2. **Neural-network demonstration:** the app includes a small fixed-weight feed-forward calculation to illustrate a multi-layer scoring architecture. It is not a trained professional model.
3. **Reinforcement-learning concept:** thumbs-up/down feedback updates a contextual-bandit preference score in local browser storage or Streamlit session state.

## Limitations and risks

- 2D angles are sensitive to camera viewpoint and perspective.
- Occlusion, loose clothing, low light, multiple people, or cropped limbs can reduce accuracy.
- Golf clubs are not explicitly detected.
- The synthetic score thresholds are educational heuristics, not validated coaching standards.
- Motion fallback cannot identify body joints and reports only motion proxies.
- Feedback may personalize suggestion ordering but does not prove that a drill is safe or effective.

## Human oversight

Users should review recommendations critically and consult a qualified golf instructor for technique changes. Stop any activity that causes pain and consult an appropriate healthcare professional when needed.
