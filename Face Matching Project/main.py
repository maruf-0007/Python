"""
# Face Matching Project
# Enable virtual environment
# Install deepface library - 'pip install deepface'

from deepface import DeepFace
result = DeepFace.verify("img1.jpg","img2.jpg")
print(result)

from deepface import DeepFace
result = DeepFace.analyze(
    img_path="img1.jpg",
    actions=["age",'gender','emotion','race'],
)
print(result)
"""


# people count in a group image

from deepface import DeepFace
result = DeepFace.analyze(
    img_path="group.jpg",
    detector_backend="retinaface",
    enforce_detection=True
)
print(len(result))
