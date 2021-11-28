# Линейная интерполяция


### Билинейная интерполяция
Реализован алгоритм билинейной интерполяции на основе шаблона байера:
![bayer masks](https://user-images.githubusercontent.com/76070534/143766444-cbbab5f3-3939-418e-b786-5b2a4661c81b.png)

### Улучшенная линейная интеполяция
Также реализован алгоритм улучшенной линейной интерполяции, используя следующую статью:
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.432.6368&rep=rep1&type=pdf

### Сравнение методов
Методы сравнивались по метрике PSNR:
![psnr](https://user-images.githubusercontent.com/76070534/143766445-7302e0ea-66c1-4873-ad84-2d4573be040a.png)
Причем результаты алгоритма улучшенной линейной интерполяции на тестовых изображениях в среднем лучше на 70%, чем результаты билинейной интерполяции.

### Интерфейс
Запуск осуществляется из командной строки со следующим форматом команд:
~~~
python main.py (command) (input_image_path) (output_image_path)
~~~
Список команд:
* bilinear
* improved
* psnr

### Юнит-тесты
Также проводилось юнит-тестирование функций:
~~~
./run.py unittest (unittest_name)
~~~
Список юнит-тестов:
* masks
* colored_img
* bilinear
* bilinear_img
* improved
* improved_img
* psnr
