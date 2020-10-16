# Using the system

To run the system simply execute the `python main.py` command supplied with the desired flags. The system supports the following list of flags:

|Settings Flag|Description|
|---|---|
|-s SETTINGS_FILE, --settings SETTINGS_FILE|The system settings file (default: ../settings.ini)|

|Execution Mode Flags|Description|
|---|---|
|-t1, --first|In this mode, the script will execute the first subtask of the challenge (default: False)|
|-t2, --second|In this mode, the script will execute the second subtask of the challenge (default: False)|

|Configuration Flags|Description|
|---|---|
|-p, --showprints|When active, some parts of the processing are shown during execution (default: False)|
|-c, --cleaning|When active, clinical reports are processed and cleaned (default: False)'|
|-m, --method|The method used to process clinical records|
|-ds, --dataset|The dataset used to process|
|-r, --read|Read annotations from file (Useful for task 2)| 
|-sf, --submission|If activated, the output will be in the submission file format|

While the system provides the possibility to define the desired method and dataset through the previous system flags, the user can also define them directly in the `settings.ini` file. By configuring `settings.ini` accordingly, it is not necessary to supply the method and dataset flags to `main.py`.

### Settings file
The PatientFM tool uses a settings file following the standard configuration file format to define the system variables. This file is divided in the following 9 sections (a description of each section is provided further below):

```ini
[vocabulary]
anatomy_file=../vocabularies/n2c2_ANAT.tsv
disease_file=../vocabularies/n2c2_DISO_extended.tsv

[methods]
deeplearning=False
rulebased=True

[embeddings]
vocabulary_path=../vocabularies/vocabulary.txt
sentences_path=../vocabularies/sentences.pickle
wordvec_path=/backup/data/models/embeddings/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin
biowordvec_original=embeddings/biowordvec_vocab_orig.npy
biowordvec_normalized=embeddings/biowordvec_vocab_norm.npy
wordvec_size=200
train_embeddings_pickle=../dataset/Train/train_embeddings.pickle
test_embeddings_pickle=../dataset/Test/test_embeddings.pickle

[datasets]
train_files=../dataset/Train/bioc-FH-training-updated-627/
test_files=../dataset/Test/testRelease-0805/
goldstandard_st1=../dataset/Train/train_subtask1_2.tsv
goldstandard_st2=../dataset/Train/train_subtask2_2.tsv
nltk_sources=../dataset/nltk_data

[neji]
use_neji_annotations=True
neji_train_pickle_biowordvec=../dataset/Train/neji_train_classes_embedmodel_BIO.pickle
neji_test_pickle_biowordvec=../dataset/Test/neji_test_classes_embedmodel_BIO.pickle
neji_train_pickle_albert=../dataset/Train/neji_train_classes_albertmodel_BIO.pickle
neji_test_pickle_albert=../dataset/Test/neji_test_classes_albertmodel_BIO.pickle
neji_train_pickle_clinicalbert=../dataset/Train/neji_train_classes_clinicalbertmodel_BIO.pickle
neji_test_pickle_clinicalbert=../dataset/Test/neji_test_classes_clinicalbertmodel_BIO.pickle

[results]
task1=../results/task1.tsv
task2=../results/task2.tsv
family_members=../results/task1.tsv
observations=../results/task1_js.tsv

[DLmodel]
#model=biowordvec_bilstm
#model=albert_bilstm
model=clinicalbert_bilstm
#model=clinicalbert_linear

[DLmodelparams]
entity_prediction=True
hiddensize=256
batchsize=32
iterationsperepoch=100
numlayers=2
learningrate=1e-3
patience=5
epochs=100
EMBEDDINGS_FREEZE_AFTER_EPOCH=2

[ALBERT]
model=albert-base-v2
#model=albert-large-v2
#model=albert-xlarge-v2
#model=albert-xxlarge-v2
add_special_tokens=True
```

| Field           | Description |
|-----------------|-------------|
|`[vocabulary]`|Section used to define vocabulary files. It is necessary to create a new entry for each new vocabulary file. These files should respect the format of the following example: ''UMLS:C0006629:T017:Anatomy    cadaver\|cadavers\|corpse\|corpses''.|
|`[methods]`|Section used to define the desired method to process clinical records. The user can select the Deep-Learning component for observation extraction, the Rule-Based component only, or set both to `True` for the Hybrid solution.|
|`[embeddings]`|Section used to define paths and parameters for word embeddings files. Not necessary if `deeplearning`is set to `False`.|
|`[datasets]`|Section used to define paths for the datasets.|
|`[neji]`|Section used to define the paths for Neji annotation files for the Deep Learning solution. This is used to avoid the process of creating Neji annotations whenever the system is run (annotations are created once and saved).|
|`[results]`|Section used to define the paths where result files will be saved.|
|`[DLmodel]`|Section used to select the Deep Learning model to use. The system currently supports 4 different models. Uncomment only the desired model.|
|`[DLmodelparams]`|Section used to set a list of Deep Learning model parameters.|
|`[ALBERT]`|Section used to select the size of the ALBERT model, if `albert_bilstm` is selected in `[DLmodel]`.|
