## Test 1: change embeddings to get_bone_length_vector() and extend fingers.
# Embedding similarity will stay the same, since bone length doesn't change when finger is extended.

from data.data_reader import DataReader
import json
from embeddings.vector import get_joint_angle_vector, get_bone_length_vector
from calculations.similarity import embedding_similarity

with open('poses/rock_on.json') as f:
    pose1 = DataReader.convert_json_to_HandPose(json_data=json.load(f))
    pose1.normalize()
    emb1 = get_bone_length_vector(pose1)

with open('poses/rock_on.json') as f:
    pose2 = DataReader.convert_json_to_HandPose(json_data=json.load(f))
    pose2.normalize()
    #pose2.straighten_finger("pinky")
    #pose2.straighten_finger("index")
    #pose2.straighten_finger("thumb")
    #pose2.straighten_finger("middle")
    #pose2.straighten_finger("ring")
    emb2 = get_bone_length_vector(pose2)

method, sim = embedding_similarity(emb1, emb2, method="mahalanobis")

print(f"{method.capitalize()} Embedding Similarity Score is {sim}")


