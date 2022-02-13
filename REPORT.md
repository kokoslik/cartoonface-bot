# Отчет по проекту
## Модели
За основу взята классическая CycleGAN из [https://arxiv.org/abs/1703.10593](https://arxiv.org/abs/1703.10593):

- [Генератор](src/model/generator.py): сеть с шестью Residual Blocks из статьи
- [Дискриминатор](src/model/discriminator.py): PatchGAN

Реализован [пул](src/utils/image_pool.py) для картинок с предыдущих эпох

В качестве Cycle и Identity лосса использовался L1Loss.

Для лоссов генератора и дискриминатора в основном использовался MSELoss, но для датасета horses2zebras также попробовал BCEWithLogitsLoss.


## Датасеты и результаты
### Датасет horses2zebras
**Цель:** превратить лошадь в зебру и обратно.

[Ссылка на скачивание датасета](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip)

Ниже представлены графики лоссов и результаты обучения на тестовой выборке для разных лоссов генератора и дискриминатора. 

Число эпох = 200, learning rate = 0.0002

|       MSELoss   |BCEWithLogitsLoss|
|-----------------|-----------------|
|![Result for MSELoss](results/horses_lsgan/199_result.jpg)|![Result for BCEWithLogitsLoss](results/horses_gan/199_result.jpg)|
|![Loss for MSELoss](results/horses_lsgan/199_loss.jpg)|![Loss for BCEWithLogitsLoss](results/horses_gan/199_loss.jpg)|

В итоге, по субъективным оценкам, вариант с MSELoss был выбран как лучший.

#### Лучшие результаты на тестовой выборке
|Horse -> Zebra|Zebra->Horse|
|--------------|------------|
|![](results/horses_lsgan/goodZebras/1.jpg)|![](results/horses_lsgan/goodHorses/1.jpg)|
|![](results/horses_lsgan/goodZebras/2.jpg)|![](results/horses_lsgan/goodHorses/2.jpg)|
|![](results/horses_lsgan/goodZebras/3.jpg)|![](results/horses_lsgan/goodHorses/3.jpg)|
|![](results/horses_lsgan/goodZebras/4.jpg)|![](results/horses_lsgan/goodHorses/4.jpg)|

#### Худшие результаты на тестовой выборке
|Horse -> Zebra|Zebra->Horse|
|--------------|------------|
|![](results/horses_lsgan/badZebras/1.jpg)|![](results/horses_lsgan/badHorses/1.jpg)|
|![](results/horses_lsgan/badZebras/2.jpg)|![](results/horses_lsgan/badHorses/2.jpg)|
|![](results/horses_lsgan/badZebras/3.jpg)|![](results/horses_lsgan/badHorses/3.jpg)|
|![](results/horses_lsgan/badZebras/4.jpg)|![](results/horses_lsgan/badHorses/4.jpg)|

[Ссылка на результаты обучения с MSELoss](https://disk.yandex.ru/d/6gAcsqjiOVxkDw)

[Ссылка на результаты обучения с BCEWithLogitsLoss](https://disk.yandex.ru/d/sZVrJr8UprHikw)

### Датасет faces2k
**Цель:** сделать лицо человека "мультяшным".
Датасет собран из двух:
1. [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), aligned&cropped
2. [Cartoon Set dataset](https://google.github.io/cartoonset/index.html)

Из каждого датасета взято по 2000 тренировочных и 200 тестовых изображений.

[Ссылка на датасет](https://disk.yandex.ru/d/PDjvBXxpyZzthA)

Ниже представлен график лоссов и результат на тестовой выборке после 200 эпох обучения с learning rate 0.0001

|Result|Loss|
|------|----|
|![Result](results/faces2k/199_result.jpg)|![Result for BCEWithLogitsLoss](results/faces2k/199_loss.jpg)|

#### Лучшие результаты на тестовой выборке
|Face -> Cartoon|Cartoon->Face|
|--------------|------------|
|![](results/faces2k/goodCartoons/1.jpg)|![](results/faces2k/goodFaces/1.jpg)|
|![](results/faces2k/goodCartoons/2.jpg)|![](results/faces2k/goodFaces/2.jpg)|
|![](results/faces2k/goodCartoons/3.jpg)|![](results/faces2k/goodFaces/3.jpg)|
|![](results/faces2k/goodCartoons/4.jpg)|![](results/faces2k/goodFaces/4.jpg)|

#### Худшие результаты на тестовой выборке
|Face -> Cartoon|Cartoon->Face|
|--------------|------------|
|![](results/faces2k/badCartoons/1.jpg)|![](results/faces2k/badFaces/1.jpg)|
|![](results/faces2k/badCartoons/2.jpg)|![](results/faces2k/badFaces/2.jpg)|
|![](results/faces2k/badCartoons/3.jpg)|![](results/faces2k/badFaces/3.jpg)|
|![](results/faces2k/badCartoons/4.jpg)|![](results/faces2k/badFaces/4.jpg)|

[Ссылка на результаты обучения](https://disk.yandex.ru/d/hupte0YPwjbo3g)

## Бот
Для удобного использования натренированной модели был написан [бот](src/bot.py) в Telegram.

Бот доступен по [ссылке](https://t.me/CartoonFaceBot).
