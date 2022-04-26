<h1> Image captioning with scene graph </h1>

<h2> Data preparation </h2>

Create a folder called 'data'

Unzip all files and place the folders in 'data' folder.

<br>

Set up the graph-rcnn.pytorch repository.

<br>
Next type this command in a python environment: 
```bash
python bottom-up_features/create_sg_h5.py
```


<br>

Although we make use of the official COCO captioning evaluation scripts, for legacy kept the nl_eval_master folder. 

Next, go to nlg_eval_master folder and type the following two commands:
```bash
pip install -e .
nlg-eval --setup
```
This will install all the files needed for evaluation. 


<h2> Training </h2>

To train the bottom-up top down model, type:
```bash
python train.py
```

<h2> Evaluation </h2>

To evaluate the model on the karpathy test split, edit the eval.py file to include the model checkpoint location and then type:
```bash
python eval.py
```

Beam search is used to generate captions during evaluation. Beam search iteratively considers the set of the k best sentences up to time t as candidates to generate sentences of size t + 1, and keeps only the resulting best k of them. A beam search of five is used for inference.

The metrics reported are ones used most often in relation to image captioning and include BLEU-4, CIDEr, METEOR and ROUGE-L. Official MSCOCO evaluation scripts are used for measuring these scores.
  
<h2>References</h2>

Code adapted with thanks from https://github.com/poojahira/image-captioning-bottom-up-top-down
