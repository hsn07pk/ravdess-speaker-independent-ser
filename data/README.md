# Data

## RAVDESS speech (not included)

The audio is the speech part of RAVDESS, `Audio_Speech_Actors_01-24.zip` from [Zenodo record 1188976](https://zenodo.org/records/1188976) (md5 `bc696df654c87fed845eb13823edef8a`): 1440 clips, 24 actors, 8 emotions. It is licensed CC BY-NC-SA 4.0 by its authors and is not redistributed here.

```bash
bash scripts/download_ravdess.sh   # downloads, checks the md5 and unzips into data/RAVDESS_speech
```

To use a copy elsewhere, set `RAVDESS_DIR` to the folder that holds `Actor_01` to `Actor_24`. `features/meta.csv` stores paths relative to that folder.

## human_ratings/

Listener ratings released with RAVDESS: the S1 to S4 Tables of Livingstone and Russo (2018), *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)*, PLoS ONE 13(5): e0196391, [doi:10.1371/journal.pone.0196391](https://doi.org/10.1371/journal.pone.0196391), licensed CC BY 4.0. See [human_ratings/SOURCE.md](human_ratings/SOURCE.md).
