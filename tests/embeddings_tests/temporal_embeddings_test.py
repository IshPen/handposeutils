import json

from handposeutils.data.data_reader import DataReader
from handposeutils.embeddings.vector import structured_temporal_embedding, flatten_temporal_embedding
from handposeutils.embeddings.vector import get_fused_pose_embedding  # embedding function, or choose angle-based fn
from handposeutils.calculations.similarity import embedding_similarity

with open('poses/rock_sequence.json') as f:
    rock_sequence1 = DataReader.convert_json_to_HandPoseSequence(json_data=json.load(f))


## EXAMPLE 1 -- get structured embedding (no PCA)
structured1 = structured_temporal_embedding(rock_sequence1, get_fused_pose_embedding, max_length=20, include_velocity=True)
print("structured shape:", structured1.shape)  # (20, per_frame_dim)

## EXAMPLE 2 -- flattened for classic classifiers
flat1 = flatten_temporal_embedding(rock_sequence1, get_fused_pose_embedding, max_length=20, include_velocity=True)
print("flat length:", flat1.shape)  # (20 * per_frame_dim,)

## EXAMPLE 3 -- variable-length structured (for RNN)
structured_var1 = structured_temporal_embedding(rock_sequence1, get_fused_pose_embedding, max_length=None, include_velocity=True)
# structured_var shape == (seq_len, per_frame_dim)
print(structured_var1)
# print(structured_var1.shape)


with open('../data_tests/saved_poses/rock_on_recording.json') as f:
    rock_sequence2 = DataReader.convert_json_to_HandPoseSequence(json_data=json.load(f))

structured2 = structured_temporal_embedding(rock_sequence2, get_fused_pose_embedding, max_length=20, include_velocity=True)
flat2 = flatten_temporal_embedding(rock_sequence2, get_fused_pose_embedding, max_length=20, include_velocity=True)

print("Same Sequence Similarity:", embedding_similarity(structured1, structured2))

with open('../data_tests/saved_poses/peace_sign_recording.json') as f:
    peace_sequence = DataReader.convert_json_to_HandPoseSequence(json_data=json.load(f))

structured_peace = structured_temporal_embedding(peace_sequence, get_fused_pose_embedding, max_length=20, include_velocity=True)
flat_peace = flatten_temporal_embedding(peace_sequence, get_fused_pose_embedding, max_length=20, include_velocity=True)

print("Different Sequence Similarity:", embedding_similarity(flat1, flat_peace))