import soundfile as sf
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf
import os
import torch
import pandas as pd
import numpy as np







class embedd_model:

  def __init__(self, processor, model, intermediate_embedding_layer = False, layer_index = False, pooling = False):
    self.processor = processor
    self.model = model
    self.pooling = pooling
    self.intermediate_embedding_layer = intermediate_embedding_layer
    self.layer_index = layer_index


  def audio_to_representation(self, audio_file_path):
      # Read the FLAC file
    data, sample_rate = sf.read(audio_file_path)

    # If the audio file has multiple channels, convert it to mono
    if len(data.shape) > 1 and data.shape[1] > 1:
        print("Speech has multiple channel")
        data = data.mean(axis=1)  # Convert to mono by averaging channels

    # Convert the audio data to a float array
    float_array = data.astype(float)

    print("Audio file is converted to float type array of shape {}".format(float_array.shape))
    input_values = self.processor(float_array, return_tensors="pt").input_values  # Batch size 1

    if self.intermediate_embedding_layer:
      try:
        hidden_state = self.model(input_values).hidden_states[self.layer_index]
      except:
        print("################# Check the hidden layer index #################")  
        exit(1)
    else:
      hidden_state = self.model(input_values).last_hidden_state


    return hidden_state


  def extract_label(self, txt_file_path):  #function to make a table of each file data labeling
    with open(txt_file_path, 'r') as file:
      lines = file.readlines()
    data = [line.split() for line in lines]
    df = pd.DataFrame(data, columns=['dummy1','file', 'dummy2', 'dummy3', 'dummy4','label', 'dummy5', 'dummy6'])
    return df


  def complete_embedding(self, directory, label_file_path, get_label = True): #by default use label data
    # List all files in the given directory
    files_in_directory = os.listdir(directory)
    recorded_audio_list = []

    # Filter out only the file names (excluding directories)
    file_names = [file for file in files_in_directory if os.path.isfile(os.path.join(current_directory, file))]

    # Display the file names
    count = 0
    df = pd.DataFrame(columns=[f"feature_{i}" for i in range(self.model.config.output_hidden_size)])

    if get_label:
      label_data = self.extract_label(label_file_path)
      self.y = pd.DataFrame(columns = ['label'])

    for file_name in file_names:
      #if count == 50:
      #  break
      count += 1
      print("id {}th file {} is processing ____________________________________".format(count, file_name))
      representation_layers = self.audio_to_representation(file_name)
      if self.pooling:
        #may try attention based pooling later
        pass
      else:
        representation_layers = torch.mean(representation_layers[0], dim=0)
      # Convert to a NumPy array and then a DataFrame row
      row = pd.DataFrame(representation_layers.detach().numpy().reshape(1, -1), columns=df.columns) #representation vectors are stored to crete X of inputt data for ML
      # Append the row to the DataFrame
      df = df.append(row, ignore_index=True)

      if get_label:
        try:
          #print("ggmu")
          label = label_data.loc[label_data['file']+'.flac' == file_name, 'label'].to_numpy().reshape(1) #search label from label data by the file name
          #print(label[0], y)
          label = pd.DataFrame(label)
          label.columns = self.y.columns
          self.y = self.y.append(label)
          recorded_audio_list.append(file_name)
          print("################# got label #################")
        except:
          print("id {}th file {} is not processed ____________________________________".format(count, file_name))
          df.drop(df.index[-1], inplace=True)  #if the label for an input file not found then ignore it
          #break

    if get_label:
      self.y.reset_index(drop=True, inplace=True)
      recorded_audio_column = pd.DataFrame(recorded_audio_list, columns = ['file_name'])
      df = pd.concat([recorded_audio_column, df, self.y], axis = 1)
    return df
  





if __name__ == "__main__":

  processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")  #code source https://huggingface.co/transformers/v4.6.0/_modules/transformers/models/wav2vec2/modeling_wav2vec2.html,  forward function
  model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", output_attentions=True, output_hidden_states=True)

  # Get the current directory
  current_directory = os.getcwd()
  
  intermediate_embedding_layer = True
  layer_index = 2
  embedded_model = embedd_model(processor, model, intermediate_embedding_layer = intermediate_embedding_layer, layer_index = layer_index)
  df = embedded_model.complete_embedding(current_directory, '/home/subhajit/ASVspoof2021_LA_eval/trial_metadata.txt')
  df.to_csv("/home/subhajit/ASVspoof2021_LA_eval/eval_intermediate_embedding_layer_"+str(intermediate_embedding_layer)+"_layer_index_"+str(layer_index)+"2021.csv", index = False)