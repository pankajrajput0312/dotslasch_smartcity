# Apartment Floor Plan Generator
Apartment Floor Plan Generator is an AI-based Desktop app that can generate multiple different kinds of floor plans



## The problem Apartment Floor Plan Generator solves
Architecture is one of the Fields, in which the current usage of AI is very less so we're initiating a process that if we use AI it will give freedom to the user to use multiple different kinds of floor plans without investing huge money in designing architectures designs from the professionals,
Currently, Apartment Design Generator is quite simple but,
1) NO introduction of AI in architecture Fields.
2) Lot's of people built their houses without consulting with the architectural engineer which leads to imperfect house structures.
3) This will provide multiple different floor plan designs to the users so the user can select best fit plans design for his house.

## How to use it,
<details open>
<summary>Install</summary>
Clone repo and install [requirements.txt]() in a
[**Python>=3.6.0**](https://www.python.org/) environment, including

```bash
https://github.com/pankajrajput0312/dotslasch_smartcity.git  # clone
cd dotslasch_smartcity
pip install -r requirements.txt  # install
```

Download weights from this [Link](https://drive.google.com/drive/folders/1ruDtsEx0feTdFYE2BtSxVGSWshppkX8G?usp=sharing)
Rename folder with name weights and place it inside dotslasch_smartcity folder

<details>
<summary>Inference with apartment_generator_final.py</summary>

```bash
python apartment_generator_final.py
```
</details>

## Challenges we ran into
We faced multiple challenges in this project,
1) our biggest challenge is, finding a dataset, there is no open-source dataset available for the floor plan design generation.
So we talked to the Walkin design studio to provide architecture furniture designs to us.
2) Even after getting furniture design problem doesn't resolve because each and every furniture designs are different, so we planned to regenerate the dataset using those floor plans designs. for that we label each and every region of furniture, then write code to make it generalize
3) Connecting the Deep Learning model with the GUI.

