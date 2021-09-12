# Text-Classification
Classifies text data into one of these 5 categories -> Beauty, Fitness, Food, Parenting, Sports.
## How to run
Dowload pretrain model from: [link](https://drive.google.com/file/d/1BHNtnOKmoR8E-m_XEbt1_c1Iyhx3Uu58/view?usp=sharing)

After unzip you will have 2 files:
- bert.ckpt
- pytorch_model.bin

Copy ```bert.ckpt``` into ```Social/saved_dict```

Copy ```pytorch_model.bin``` into ```bert_pretrain```
```
pip install -r requirements.txt
```

To run with single sentence

```python predict.py --text "Gummies that shrink your waist AND tastes good😍!?!?! I got you 😉 link in bio🎉#linkinbio #itworks #slimminggummies #snatched #slimwaist #summerready #summertimefine #hotgirl #gummies #flattummy #wellness #health #fitness #motivation #bodygoals #body"```

To run with csv file

```python predict.py --csv <path_to_csv>```
