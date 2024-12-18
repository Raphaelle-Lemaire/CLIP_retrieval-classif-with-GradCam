Pour lancer le code, aller dans LAVIS et tapper: " python -m torch.distributed.run train.py --cfg-path lavis/projects/clip/train/ret_artpedia.yaml ".
Ce code est tiré de la bibliothèque LAVIS auquel ont à réimplémenté un GRADCAM à retrouver dans le jupyterNotebook " LAVIS/CompletGradCam.ipynb ".
Nous avons aussi rajouter une couche de classification sur les transformers de textes (actuellement ne fonctionne pas)
