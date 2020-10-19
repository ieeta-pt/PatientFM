<<<<<<< HEAD
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
=======
<h1 align="center"><b>PatientFM</b></h1>

<p align="center"><b>An end-to-end system for extracting family history information from clinical notes.</b></p>

### Documentation

PatientFM is an end-to-end hybrid solution composed of a Rule-based Engine that integrates Deep Learning approaches for entity extraction.
For more documentation on how to use the system, please check the instructions provided [here](https://github.com/bioinformatics-ua/PatientFM/blob/master/src/README.md).

### Team
  * João F. Silva<sup id="a1">[1](#f1)</sup>
  * João R. Almeida<sup id="a1">[1](#f1)</sup><sup id="a2">[2](#f2)</sup>
  * Sérgio Matos<sup id="a1">[1](#f1)</sup>

1. <small id="f1"> University of Aveiro, Dept. Electronics, Telecommunications and Informatics (DETI / IEETA), Aveiro, Portugal </small> [↩](#a1)
2. <small id="f4"> University of A Coruña, Dept. of Information and Communications Technologies, A Coruña, Spain </small> [↩](#a4)

### Cite

Please cite the following, if you use PatientFM in your work:

```bib
in progress
```
>>>>>>> 0a0d462925f1f6a6316b426487dd0f64421c8fbb
