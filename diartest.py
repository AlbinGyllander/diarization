

import torch
from speechbrain.pretrained import SpeakerRecognition
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import librosa
import numpy as np

def diarize_audio(audio, model, max_num_speakers=10, segment_len=1.0, overlap=0.1,sr=16000):
    # Load the audio file
    
    
    # Segment the audio with overlap
    step = int((1 - overlap) * segment_len * sr)
    segments = [audio[i:i + int(segment_len * sr)] for i in range(0, len(audio) - int(segment_len * sr), step)]
    
    # Extract d-vectors for each segment
    d_vectors = []
    for segment in segments:
        embedding = model.encode_batch(torch.tensor(np.array([segment])))
        d_vectors.append(embedding.squeeze().numpy())
    
    # Compute the optimal number of clusters
    best_num_clusters = 1
    best_silhouette_score = -1
    for num_clusters in range(2, min(max_num_speakers, len(d_vectors)) + 1):
        clustering = AgglomerativeClustering(n_clusters=num_clusters).fit(d_vectors)
        labels = clustering.labels_
        score = silhouette_score(d_vectors, labels)
        if score > best_silhouette_score:
            best_silhouette_score = score
            best_num_clusters = num_clusters
    
    # Final clustering with best_num_clusters
    clustering = AgglomerativeClustering(n_clusters=best_num_clusters).fit(d_vectors)
    labels = clustering.labels_
    
    def merge_segments(segments):
        merged_segments = []
        start_time, end_time, prev_label = segments[0]
        for st, et, label in segments[1:]:
            if label == prev_label:
                end_time = et  # extend the segment
            else:
                merged_segments.append((start_time, end_time, prev_label))
                start_time, end_time, prev_label = st, et, label  # start a new segment
        merged_segments.append((start_time, end_time, prev_label))
        return merged_segments
    
    segments = [(i * step / sr, (i + 1) * step / sr, label) for i, label in enumerate(labels)]
    
    # First Merge
    segments = merge_segments(segments)
    
    # Filter
    segments = [(st, et, label) for st, et, label in segments if et - st >= 1]
        
    # Second Merge
    segments = merge_segments(segments)
    
    
    # Convert to MM:SS format
    final_segments = []
    for start_time, end_time, label in segments:
        start_str = f"{int(start_time // 60):02d}:{int(start_time % 60):02d}"
        end_str = f"{int(end_time // 60):02d}:{int(end_time % 60):02d}"
        final_segments.append((start_str, end_str, label))
    
    return final_segments

if __name__ == '__main__':
    audio, sr = librosa.load("meeting.wav", sr=16000)
    
    model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir_ecapa")

    diag = diarize_audio(audio=audio,model=model,max_num_speakers=4)
    print(diag)
