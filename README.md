# PatientFM

to do

### Settings file

PatientFM tool uses a settings file following the standard configuration file format to define the system variables. The vocabularies used in the processing were defined in the ''vocabulary'' section. In this section is created a new entry for each new vocabulary file. These files should respect the format of the following example: ''UMLS:C0006629:T017:Anatomy    cadaver|cadavers|corpse|corpses''.
The datasets section contains the location of the dataset to process and the gold standard files for each task. Finally, the results section defines the path to write the system execution output.

```ini
[vocabulary]
anatomy_file=../vocabularies/n2c2_ANAT.tsv
disease_file=../vocabularies/n2c2_DISO_extended.tsv
...

[datasets]
;Change the paths here to execute with the test set
files=../dataset/Test/testRelease-0805/
goldstandard_st1=../dataset/Train/train_subtask1_2.tsv
goldstandard_st2=../dataset/Train/train_subtask2_2.tsv

[results]
task1=../results/task1.tsv
task2=../results/task2.tsv
```