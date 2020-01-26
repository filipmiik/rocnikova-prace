# Praktická část k ročníkové práci
Tento projekt demonstruje způsob využití a aplikace rozpoznání obličejů za využití knihovny OpenCV a předtrénovaného klasifikátoru, který využívá Haarových vlnek.

## Princip
Program při spuštění načte dostupné obrázky ze složky `images/_source`, zpracuje je a nalezené obličeje po jednom uloží do složky `images/faces`.

Pro extrahování obličejů z obrázků, nahrajte fotografie do složky `images/_source` a spusťte příkaz

    $ python main.py

## Využité knihovny
1. `opencv-python` (OpenCV upravené pro Python)
2. `numpy`

##  Využité klasifikátory
1. `haarcascade_frontalface_alt.xml` dostupný z repozitáře `opencv/opencv/data/haarcascades`
