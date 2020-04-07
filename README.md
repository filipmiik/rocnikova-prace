# Praktická část k ročníkové práci
Tento projekt demonstruje způsob využití a aplikace rozpoznání obličejů za využití knihovny OpenCV a předtrénovaného klasifikátoru, který využívá Haarových vlnek.

## Princip
Program při spuštění načte dostupné obrázky ze složky `images/_source`, zpracuje je a nalezené obličeje po jednom uloží do složky `images/faces`.

Pro extrahování obličejů z obrázků nahrajte fotografie do složky `images/_source` a spusťte příkaz

    $ python main.py

## Využité knihovny
1. `opencv-python` (OpenCV upravené pro Python)
2. `numpy`

##  Využité klasifikátory
1. `haarcascade_frontalface_alt.xml` dostupný z repozitáře `opencv/opencv/data/haarcascades`

## Využité obrázky
1. https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/2017_class_of_NASA_astronauts_in_January_2020.jpg/1280px-2017_class_of_NASA_astronauts_in_January_2020.jpg
2. https://pixabay.com/get/5fe1dd474d53b108feda84608229347d143ddae0564c704c7d2e78d2944bc65a/pedestrians-918471_1920.jpg?attachment
3. https://pixabay.com/get/57e2d1464d55aa14f6d1867dda6d367d163edce156546c4870277ad7914ec05ab0/workplace-1245776_1920.jpg?attachment
4. https://pixabay.com/get/57e7d14b484fad00f5d8997cc42b367c113adbf85254794f772779d09648_1920.jpg
5. https://pixabay.com/get/57e2d54a4c55ad14f6d1867dda29347d143ddae0564c704c7d2979dc9548c55c_1920.jpg
6. https://pixabay.com/get/57e9d24a4854ad14f6d1867dda29347d143ddae0564c704c7d2979dc9548cd51_1920.jpg
