## Structure

- data/: data loaders etc.
- log/: checkpoints, runtime logs etc.
- config/: execution configurations
- model/: model definitions
- util/: scheduler, spectrogram extractor, etc.

## Prepare

```
conda create -n <env_name>
conda activate <env_name>

pip install torch torchvision torchaudio

pip install -r requirements.txt
```

Download TAU Urban Acoustic Scenes 2022 Mobile dataset [here](https://zenodo.org/records/6337421)

Download Microphone Impulse Response dataset [here](https://micirp.blogspot.com/?m=1)

Dataset should be put in parent folder of current project folder, you should have folder structure

- ../TAU-urban-acoustic-scenes-2022-mobile-development/development/audio/: 230350 audio files
- ../microphone_impulse_response/: 67 audio files

## Fine-tuning

1. BEATs (SSL\*)

```
python main.py fit --config config/beats_ssl_star.yaml
```

2. BEATs (SSL)

```
python main.py fit --config config/beats_ssl.yaml
```

3. BEATs (SSL+SL)

```
python main.py fit --config config/beats_ssl+sl.yaml
```

4. Testing

```
python main.py test --config config/beats_test.yaml
```

5. Generate fine-tuned logits

```
python main.py predict --config config/beats_predict.yaml
```

## Knowledge Distillation

Specify logits file location in `tfsepnet_kd.yaml` and run

```
python main.py fit --config config/tfsepnet_kd.yaml
```

## Specify Parameters

Parameters can be specified using CLI as follows:

```
python main.py fit --config config/tfsepnet_train.yaml --trainer.max_epochs 30
```

## LEAF

Change `spec_extractor` config in config files to `util.Leaf` and `util.LeafBeats`
