import librosa
import whisper
import os 
import json
import wave
import tempfile
import librosa
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.mixture import *
import json

from sklearn.cluster import KMeans
from python_speech_features import mfcc



def VoiceActivityDetection(wavData, frameRate):
    # uses the librosa library to compute short-term energy
    ste = librosa.feature.rms(y= wavData,hop_length=int(16000/frameRate)).T
    thresh = 0.1*(np.percentile(ste,97.5) + 9*np.percentile(ste,2.5))    # Trim 5% off and set threshold as 0.1x of the ste range
    return (ste>thresh).astype('bool')

def trainGMM(wavFile, frameRate, segLen, vad, numMix):
    wavData,_ = librosa.load(wavFile,sr=16000)
    mfcc = librosa.feature.mfcc(y = wavData, sr=16000, n_mfcc=25,hop_length=int(16000/frameRate)).T
    vad = np.reshape(vad,(len(vad),))
    if mfcc.shape[0] > vad.shape[0]:
        vad = np.hstack((vad,np.zeros(mfcc.shape[0] - vad.shape[0]).astype('bool'))).astype('bool')
    elif mfcc.shape[0] < vad.shape[0]:
        vad = vad[:mfcc.shape[0]]
    mfcc = mfcc[vad,:]
    GMM = GaussianMixture(n_components=numMix,reg_covar=0.000001,covariance_type='diag').fit(mfcc)
    var_floor = 1e-5
    segLikes = []
    segSize = frameRate*segLen
    for segI in range(int(np.ceil(float(mfcc.shape[0])/(frameRate*segLen)))):
        
        startI = segI*segSize
        endI = (segI+1)*segSize
        if endI > mfcc.shape[0]:
            endI = mfcc.shape[0]-1
        if endI==startI:    # Reached the end of file
            break
        seg = mfcc[startI:endI,:]
        compLikes = np.sum(GMM.predict_proba(seg),0)
        segLikes.append(compLikes/seg.shape[0])

    return np.asarray(segLikes)



def SegmentFrame(clust, segLen, frameRate, numFrames):
    frameClust = np.zeros(numFrames)
    for clustI in range(len(clust)-1):
        frameClust[clustI*segLen*frameRate:(clustI+1)*segLen*frameRate] = clust[clustI]*np.ones(segLen*frameRate)
    frameClust[(clustI+1)*segLen*frameRate:] = clust[clustI+1]*np.ones(numFrames-(clustI+1)*segLen*frameRate)
    return frameClust

def process_temp_wav(input_wav, start_time, end_time,model,speaker):
    # Validate the input start_time and end_time
    if start_time >= end_time:
        raise ValueError("Start time must be less than end time.")

    # Create a temporary directory to store the temporary WAV file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Read the input WAV file
        with wave.open(input_wav, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            duration = n_frames / float(sample_rate)

            # Convert start and end time to frame indices
            start_frame = int(start_time * sample_rate)
            end_frame = int(end_time * sample_rate)

            # Ensure start and end frames are within valid ranges
            start_frame = max(0, min(start_frame, n_frames))
            end_frame = max(0, min(end_frame, n_frames))

            # Set the read range
            wav_file.setpos(start_frame)
            frames_to_read = end_frame - start_frame

            # Read the audio frames
            audio_frames = wav_file.readframes(frames_to_read)

        # Create a temporary WAV file inside the temporary directory
        temp_wav_file = os.path.join(temp_dir, "temp.wav")
        with wave.open(temp_wav_file, 'wb') as output_wav:
            output_wav.setnchannels(n_channels)
            output_wav.setsampwidth(sample_width)
            output_wav.setframerate(sample_rate)
            output_wav.writeframes(audio_frames)

        # Perform your operations on the temporary WAV file here
        # For example, you can call another function that processes the temp WAV file
        data = process_file(temp_wav_file,model,start=start_time,end=end_time,speaker=speaker)
        # Delete the temporary WAV file
        os.remove(temp_wav_file)

        return data


def process_file(input_wav,model,start=0,end=0,speaker=""):
    result = model.transcribe(input_wav)
    sentences = []

    for segment in result["segments"]:
        start_time = segment["start"] + float(start)
        end_time = segment["end"] + float(start)
        text = segment["text"]

        sentence = {
            "start": str(start_time),
            "end": str(end_time),
            "text": text,
            "speaker": speaker
        }
        sentences.append(sentence)

    return sentences


# This code is used to convert the speaker labels from a list of speaker labels to a list of speaker segments, where each segment has a start time, end time, and speaker label.
# The inputs are the speaker labels, the frame rate, and the name of the wav file.
# The output is a list of speaker segments, where each speaker segment is a tuple containing the speaker label, start time, and end time.

def speakerdiarisation_list(hyp, frameRate, wavFile):
    diarization_segments = []
    spkrChangePoints = np.where(hyp[:-1] != hyp[1:])[0]

    if len(spkrChangePoints) > 0:
        if spkrChangePoints[0] != 0 and hyp[0] != -1:
            spkrChangePoints = np.concatenate(([0], spkrChangePoints))

        spkrLabels = []
        for spkrHomoSegI in range(len(spkrChangePoints)):
            spkrLabels.append(hyp[spkrChangePoints[spkrHomoSegI] + 1])

        current_speaker = None
        segment_start = None
        segment_end = None

        for spkrI, spkr in enumerate(spkrLabels):
            if spkr != -1:
                if current_speaker is None:
                    current_speaker = spkr
                    segment_start = (spkrChangePoints[spkrI] + 1) / float(frameRate)
                elif current_speaker != spkr:
                    segment_end = spkrChangePoints[spkrI] / float(frameRate)
                    diarization_segments.append(("Speaker " + str(int(current_speaker)), segment_start, segment_end))
                    current_speaker = spkr
                    segment_start = (spkrChangePoints[spkrI] + 1) / float(frameRate)

        # Add the last segment if there is one
        if current_speaker is not None:
            segment_end = len(hyp) / float(frameRate)
            diarization_segments.append(("Speaker " + str(int(current_speaker)), segment_start, segment_end))

    return diarization_segments


# This code takes in the waveform data, the frame rate, and the VAD information.
# It returns the MFCC features and the VAD information, but the VAD information is now the length of the MFCC features. 
def getVad_and_mfcc(wavData,frameRate,vad):
    mfcc = librosa.feature.mfcc(y = wavData, sr=16000, n_mfcc=30,hop_length=int(16000/frameRate)).T
    vad = np.reshape(vad,(len(vad),))
    if mfcc.shape[0] > vad.shape[0]:
        print(4.1)
        vad = np.hstack((vad,np.zeros(mfcc.shape[0] - vad.shape[0]).astype('bool'))).astype('bool')
    elif mfcc.shape[0] < vad.shape[0]:
        print(4.2)
        vad = vad[:mfcc.shape[0]]
    
    return mfcc[vad,:],vad

# def get_cluster(wavFile,frameRate,segLen,vad,numMix,mfcc,numberOfClusters):
#     clusterset = trainGMM(wavFile, frameRate, segLen, vad, numMix)

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(clusterset)  
#     # Normalizing the data so that the data approximately 
#     # follows a Gaussian distribution
#     X_normalized = normalize(X_scaled)
#     cluster = AgglomerativeClustering(n_clusters=numberOfClusters, metric='euclidean', linkage='ward') 
#     clust=cluster.fit_predict(X_normalized)

#     return SegmentFrame(clust, segLen, frameRate, mfcc.shape[0])

def get_cluster(audio, num_speakers):
    # Extract MFCC features from the audio
    mfcc_feat = mfcc(audio, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512, preemph=0.97)
    
    # Cluster the MFCC features using K-means
    kmeans = KMeans(n_clusters=num_speakers, init='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=0)
    kmeans.fit(mfcc_feat)
    
    # Return the cluster labels as a list
    return kmeans.labels_.tolist()


def getBars(fileData,sr):# Step 2: Define the number of bars
    num_bars = 300
    # Step 1: Calculate the total duration of the audio
    total_duration = len(fileData) / sr

    # Step 2: Calculate the size of each bar
    bar_width = total_duration / num_bars

    # Step 3: Split the audio into segments
    segments = [fileData[int(i * bar_width * sr):int((i + 1) * bar_width * sr)] for i in range(num_bars)]

    # Step 4: Calculate the volume for each segment
    volumes = [float(np.sqrt(np.mean(segment ** 2))) for segment in segments]

    # Step 5: Downsample the data for better distinctiveness
    downsampled_volumes = downsample(volumes, num_bars)

    # Step 6: Prepare the JSON object
    jsonData = []
    for i, volume in enumerate(downsampled_volumes):
        center_x = (i + 0.5) * bar_width  # Calculate centered x-value
        data = {'x': center_x, 'y': volume}
        jsonData.append(data)

    # Step 7: Convert the data to JSON
    json_data = json.dumps(jsonData, ensure_ascii=False)

    # Print or use the JSON object as needed
    return json_data

def downsample(data, num_bars):
    # Determine the number of data points to average in each downsampled bar
    step = max(len(data) // num_bars, 1)

    # Calculate the downsampled values by averaging the data points within each step
    downsampled_data = [np.mean(data[i:i+step]) for i in range(0, len(data), step)]

    return downsampled_data



def count_sentences(text):
    return text.strip().count('.') + text.strip().count('?')

def count_sentences(text):
    result = 0
    for word in text.split():
        if word.endswith("."):
            result += 1
        elif word.endswith("?"):
            result += 1
        elif word.endswith("!"):
            result += 1
        elif word.endswith("..."):
            result += 1
    return result

def concatenate_sentences(json_data, max_sentence_length):
    data =json_data
    result = []

    current_line = []
    current_length = 0
    current_speaker = None

    for entry in data:
        text = entry["text"].strip()
        speaker = entry["speaker"]

        if current_speaker is not None and current_speaker != speaker:
            result.append({
                "start": current_line[0]["start"],
                "end": current_line[-1]["end"],
                "text": " ".join(entry["text"] for entry in current_line),
                "speaker": current_speaker
            })
            current_line = []
            current_length = 0

        sentence_count = count_sentences(text)
        if current_length + sentence_count <= max_sentence_length:
            current_line.append(entry)
            current_length += sentence_count
        else:
            result.append({
                "start": current_line[0]["start"],
                "end": current_line[-1]["end"],
                "text": " ".join(entry["text"] for entry in current_line),
                "speaker": current_speaker
            })
            current_line = [entry]
            current_length = sentence_count

        current_speaker = speaker

    if current_line:
        result.append({
            "start": current_line[0]["start"],
            "end": current_line[-1]["end"],
            "text": " ".join(entry["text"] for entry in current_line),
            "speaker": current_speaker
        })

    return json.dumps(result, indent=0,ensure_ascii=False)







class CustomTranscriptionError(Exception):
    def __init__(self, message):
        print("Error transcribing, retrying...")
        super().__init__(message)
        self.message = message



def diar(audio, num_speakers):
    # Set the parameters for diarization
    segLen, frameRate, numMix = 2, 50, 256
    
    # Compute the number of frames in each segment
    segFrames = int(segLen * frameRate)
    
    # Compute the number of segments in the audio
    numSegs = int(np.ceil(len(audio) / (segFrames * frameRate)))
    
    # Pad the audio with zeros to make it evenly divisible into segments
    audio = np.pad(audio, (0, numSegs * segFrames * frameRate - len(audio)), 'constant')
    
    # Diarize the audio using the get_cluster function
    labels = get_cluster(audio, num_speakers)
    
    # Group adjacent segments with the same label into a single speech segment
    segments = []
    start_time = 0
    prev_label = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != prev_label:
            end_time = i * segLen
            segments.append((start_time, end_time, prev_label))
            start_time = end_time
            prev_label = labels[i]
    end_time = len(labels) * segLen
    segments.append((start_time, end_time, prev_label))
    
    # Return the speech segments
    return segments
   
def process():
    file = "btmn.wav"

    segLen,frameRate,numMix = 10,50,256
    wavData,sr = librosa.load(file,sr=16000) 

    try:

        vad=VoiceActivityDetection(wavData,frameRate)
        mfcc,vad = getVad_and_mfcc(wavData,frameRate,vad) 
        
    except:
        print("cannot get Voice Activity Detection or audio features.")

    speakers_with_timesstamps = diar(wavData,2)



    try:

        json_list = []
        model = whisper.load_model("medium")
        print(speakers_with_timesstamps)
        for i in range(len(speakers_with_timesstamps)):
            
            for sentence in process_temp_wav(model=model,input_wav=file,start_time=float(speakers_with_timesstamps[i][0]),end_time=float(speakers_with_timesstamps[i][1]),speaker=speakers_with_timesstamps[i][2]):
                json_list.append(sentence)

        resulting_transcribing =  concatenate_sentences(json_list,1000)

        # create a list of speakers with unique initials
        #This code is used to find the names of the speakers in the audio and put them into a list as a dictionary. This allows for the speakers to be named and identified in the transcribing. 
        try:
            temp_speaker_list = []
            for line in json.loads(resulting_transcribing):
                if not line["speaker"] in temp_speaker_list:
                    temp_speaker_list.append(line["speaker"])
            speakers = []
            final_speaker_list = []
            for i in range(len(temp_speaker_list)):
                name = temp_speaker_list[i]
                speakerString = {"name":str(name),"initials":"S" + str(i)}
                final_speaker_list.append(speakerString)
            speakers = final_speaker_list
        except:
            speakers = [{"name":"speaker 0","initials":"S0"}]
        
        speakers = json.dumps(speakers)
        print(resulting_transcribing)
        print(speakers)

  

    except Exception as E:
        print("Enrecoverable transcription error")
        print("Reson:" + str(E))
        return 




if __name__ == '__main__':
    process()